from imagetra.common.base import Base
from imagetra.common.media import Image
from imagetra.common.logger import get_logger
from imagetra.common.bbox import Bbox
from imagetra.common.transform import transform
from imagetra.common.batch import get_batch, unflatten
from imagetra.common.result import Result

from typing import List
import numpy as np
import lap
from pyfoma import FST, State
from collections import Counter

logger = get_logger('Detector')

def pad_imags(imgs: List[Image]):
    max_width, max_height = 0, 0
    for img in imgs:
        if img.width > max_width:
            max_width = img.width
        if img.height > max_height:
            max_height = img.height
    return [
        img.pad(max_width-img.width, max_height-img.height) 
        for img in imgs
    ]

class BaseDetector(Base):
    def __init__(self, margin: float=0, padding: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.margin = margin
        self.padding = padding
    
    def forward(self, imgs: List[Image]) -> List[Result]:
        assert(len(imgs) > 0)

        if self.padding:
            imgs = pad_imags(imgs)

        results = []
        for batch in get_batch(imgs, show_pbar=self.show_pbar):
            batch_results = self.detect(batch)
            results += batch_results
        
        assert(len(results) == len(imgs))
        assert(isinstance(results[0].bboxs, list))

        if self.margin != 0:
            for result in results:
                result.bboxs = [
                    bbox.arrange().add_margin(self.margin) 
                    for bbox in result.bboxs
                ]
        return results

    def detect(self, imgs: List[Image]) -> List[Result]:
        raise NotImplementedError()

class BaseRecognizer(Base):
    def __init__(self, padding: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.padding = padding
    
    def forward(self, imgs: List[Image]) -> Result:
        assert(len(imgs) > 0)

        if self.padding:
            imgs = pad_imags(imgs)

        recognized_texts, recognized_scores = [], []
        for batch in get_batch(imgs, show_pbar=self.show_pbar):
            batch_result = self.recognize(batch)
            recognized_texts += batch_result.ocr_texts
            recognized_scores += batch_result.ocr_scores
        return Result(
            ocr_texts=recognized_texts,
            ocr_scores=recognized_scores
        )

    def recognize(self, imgs: List[Image]) -> Result:
        raise NotImplementedError()

class BaseRecoDetector(Base):
    def __init__(self, margin: float=0, padding: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.margin = margin
        self.padding = padding
    
    def forward(self, imgs: List[Image]) -> List[Result]:
        assert(len(imgs) > 0)

        if self.padding:
            imgs = pad_imags(imgs)

        results = []
        for batch in get_batch(imgs, show_pbar=self.show_pbar):
            batch_results = self.recodetect(batch)
            results += batch_results
        
        assert(len(results) == len(imgs)), (len(results), len(imgs))
        assert(isinstance(results[0].bboxs, list))

        if self.margin != 0:
            for result in results:
                result.bboxs = [
                    bbox.arrange().add_margin(self.margin) 
                    for bbox in result.bboxs
                ]
        return results

    def recodetect(self, imgs: List[Image]) -> List[Result]:
        raise NotImplementedError()

class MergeRecoDetector(BaseRecoDetector):
    def __init__(
            self,
            recognizer: BaseRecognizer, 
            detector: BaseDetector, 
            transform: bool=False,
            subbatch_size: int=10,
        ) -> None:
        self.recognizer = recognizer
        self.detector = detector
        self.transform = transform
        self.subbatch_size = subbatch_size
        super().__init__(margin=self.detector.margin, padding=None)
    
    def crop(self, img: Image, bbox: Bbox) -> Image:
        rec_bbox = bbox.to_outbound_rectangle()
        if self.transform:
            return transform(img, bbox, rec_bbox)
        return img.crop(rec_bbox.topleft, rec_bbox.bottomright)

    def forward(self, imgs: List[Image]):
        results = self.detector(imgs)
        cropped_imgs = [
            self.crop(img, bbox)
            for img, result in zip(imgs, results)
            for bbox in result.bboxs
        ]
        batch_detected_bboxs = [
            result.bboxs
            for result in results
        ]

        recognizer_result = self.recognizer(cropped_imgs)
        batch_recognized_texts = unflatten(recognizer_result.ocr_texts, batch_detected_bboxs)
        batch_recognized_scores = unflatten(recognizer_result.ocr_scores, batch_detected_bboxs)
        for result, recognized_texts, recognized_scores in zip(results, batch_recognized_texts, batch_recognized_scores):
            assert(result.ocr_texts == result.ocr_scores == None)
            result.ocr_texts = recognized_texts
            result.ocr_scores = recognized_scores
        return results

class EnsembleRecoDetector(BaseRecoDetector):
    def __init__(self, recodetectors: list, iou_theshold=0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.recodetectors = recodetectors
        self.iou_theshold = iou_theshold
        assert(len(self.recodetectors) > 0)

    def to(self, device: str):
        for recodetector in self.recodetectors:
            recodetector.to(device)

    def group_bboxs(self, bboxs1, bboxs2) -> List[Bbox]:
        row, col = len(bboxs1), len(bboxs2)
        ious = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                score = max(
                    bboxs1[i].ap(bboxs2[j]),
                    bboxs1[i].ar(bboxs2[j])
                )
                # [TODO] consider text similarity
                ious[i, j] = score
        
        costs = 1 - np.array(ious)
        _, x, y = lap.lapjv(costs, extend_cost=True)
        filtered_xy = np.argwhere(costs >= self.iou_theshold).tolist()
        group_ids = []

        # merge overlapped bboxs
        for i, j in enumerate(x):
            if [i,j] in filtered_xy:
                group_ids.append((i, -1))
                group_ids.append((-1, j))
                continue
            group_ids.append((i,j))

        # add non-overlapped bboxs
        for j, i in enumerate(y):
            if i > -1:
                continue
            if [i,j] in filtered_xy:
                group_ids.append((i, -1))
                group_ids.append((-1, j))
                continue
            group_ids.append((i,j))
        
        # assert(Counter([i for i, _ in group_ids]).most_common(2)[-1][1] == 1), Counter([i for i, _ in group_ids])
        # assert(Counter([j for _, j in group_ids]).most_common(2)[-1][1] == 1), Counter([j for _, j in group_ids])
        return group_ids

    def merge_bboxs(self, bboxs1: List[Bbox], bboxs2: List[Bbox], group_ids: List[int]) -> List[Bbox]:
        bboxs = []
        for i, j in group_ids:
            assert(i != -1 or j != -1)
            if i == -1:
                bboxs.append(bboxs2[j])
            elif j == -1:
                bboxs.append(bboxs1[i])
            elif i != -1 and j != -1:
                bboxs.append(bboxs1[i].merge(bboxs2[j]))
            else:
                raise ValueError()
        return bboxs

    def integrate_list(self, main_list: list, aux_list: list, group_ids: List[int]) -> list:
        # main_list: a list of lists, e.g., [[a], [b,c], [d]]
        # aux_list: a list of elements, e.g., [a, e, f]
        output = []
        for i, j in group_ids:
            assert(i != -1 or j != -1)
            if i == -1:
                output.append([aux_list[j]])
            elif j == -1:
                output.append(main_list[i])
            elif i != -1 and j != -1:
                main_list[i].append(aux_list[j]) # add element in aux_list to main_list
                output.append(main_list[i])
            else:
                raise ValueError()
        return output

    def merge_texts(self, texts: List[str], scores: List[float], word_level: bool=True):
        if len(set(texts)) == 1:
            return texts[0], max(scores)

        if word_level:
            texts = [text.split(' ') for text in texts]

        data = Counter()
        max_len = 0
        for text, score in zip(texts, scores):
            lenght = len(text)
            max_len = max(max_len, lenght)
            for i in range(lenght):
                key = (text[i], i, i+1)
                data[key] += score

        fst = FST()
        states = [fst.initialstate] + [State() for _ in range(max_len)]
        alphabets = set()
        for (token, start, end), freq in data.items():
            states[start].add_transition(states[end], (token, token), float(1/max(freq, 0.0001)))
            alphabets.add(token)
        fst.states = states
        fst.alphabet = alphabets
        finalstates = []
        for text in texts:
            len_text = len(text)
            states[len_text].finalweight = 0
            finalstates.append(states[len_text])
        fst.finalstates = finalstates
        score, text = fst.words_nbest(1)[0]
        delimiter = ' ' if word_level else ''
        text = delimiter.join([token for token, _ in text])
        return text, score
    
    def recodetect(self, imgs: List[Image]):
        anchor_results = self.recodetectors[0].recodetect(imgs)
        anchor_results_length = len(anchor_results)

        final_bboxs, final_ocr_texts, final_scores = [], [], []

        # initialize final_* with anchor_*
        for result in anchor_results:
            final_bboxs.append(result.bboxs.copy())
            final_ocr_texts.append([[ele] for ele in result.ocr_texts])
            final_scores.append([
                [(ele1+ele2)/2] 
                for ele1, ele2 in zip(result.ocr_scores, result.bbox_scores)
            ])

        # combine results from other recodetectors with those in final_*
        for recodetector in self.recodetectors[1:]:
            results = recodetector.recodetect(imgs)
            assert(len(results) == anchor_results_length)

            for i in range(anchor_results_length):
                group_ids = self.group_bboxs(
                    final_bboxs[i], 
                    results[i].bboxs
                )
                # merge bboxes
                final_bboxs[i] = self.merge_bboxs(final_bboxs[i], results[i].bboxs, group_ids)
                final_ocr_texts[i] = self.integrate_list(
                    final_ocr_texts[i], 
                    results[i].ocr_texts, 
                    group_ids
                )
                scores = [
                    (ele1+ele2)/2
                    for ele1, ele2 in zip(results[i].ocr_scores, results[i].bbox_scores)
                ]
                final_scores[i] = self.integrate_list(final_scores[i], scores, group_ids)
        
        # merge ocr texts
        merged_ocr_texts, merged_scores = [], []
        for i in range(anchor_results_length):
            assert(len(final_ocr_texts[i]) == len(final_scores[i]))
            merged_ocr_texts, merged_scores = [], []
            for ocr_texts, scores in zip(final_ocr_texts[i], final_scores[i]):
                merged_ocr_text, merged_score = self.merge_texts(ocr_texts, scores)
                merged_ocr_texts.append(merged_ocr_text)
                merged_scores.append(merged_score)
            
            anchor_results[i].bboxs = final_bboxs[i]
            anchor_results[i].ocr_texts = merged_ocr_texts
            anchor_results[i].ocr_scores = merged_scores
            anchor_results[i].bbox_scores = merged_scores

        return anchor_results
