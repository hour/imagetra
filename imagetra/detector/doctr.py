from imagetra.common.logger import get_logger
logger = get_logger('DoctrRecoDetector')

try:
    from doctr.io.elements import Page, Block
except ModuleNotFoundError as e:
    logger.error("The 'doctr' package is not installed. Install with `pip install imagetra[doctr]`")
    raise

from imagetra.detector.base import BaseDetector, BaseRecognizer, BaseRecoDetector
from imagetra.common.media import Image
from imagetra.common.bbox import Bbox, Coordinate
from imagetra.common.result import Result

from typing import List
import torch
import numpy as np

# methods
# - FAST: https://github.com/czczup/FAST?tab=readme-ov-file
# - EAST: https://github.com/argman/EAST
# - tensorflow_PSENet: https://github.com/liuheng92/tensorflow_PSENet

class DoctrDetector(BaseDetector):
    # check doctr for model name
    # e.g., db_resnet50, fast_base
    def __init__(self, name='fast_base', assume_straight_pages=False, **kwargs) -> None:
        super().__init__(**kwargs)
        from doctr.models import detection

        # self.model = detection.fast_tiny(pretrained=True).eval()
        self.model = detection.detection_predictor(
            name,
            pretrained=True,
            preserve_aspect_ratio=False,
            symmetric_pad=True,
            assume_straight_pages=assume_straight_pages, # [TODO] this should be false because of scense image
        )

    def to(self, device: str):
        self.model.to(device)

    def detect(self, imgs: List[Image]) -> List[Result]:
        def _format_output(words):
            bboxs, scores = [], []
            for word in words:
                assert(word.shape == (5,2))
                bboxs.append(
                    Bbox(
                        topleft=Coordinate(*word[0]),
                        topright=Coordinate(*word[1]),
                        bottomright=Coordinate(*word[2]),
                        bottomleft=Coordinate(*word[3])
                    )
                )
                scores.append(word[4][1])
            return bboxs, scores                
        
        imgs_tensor = torch.stack([img.to_channel_first().to_tensor() for img in imgs])
        _, _, height, width = imgs_tensor.shape

        outs = self.model(imgs_tensor)

        # Convert to absolute coordinates
        for batch_id in range(len(outs)):
            outs[batch_id]['words'][:, :-1, 0] *= width
            outs[batch_id]['words'][:, :-1, 1] *= height
        
        results = []
        for out in outs:
            bboxs, scores = _format_output(out['words'])
            results.append(Result(bboxs=bboxs, bbox_scores=scores))
        return results

class DoctrRecognizer(BaseRecognizer):
    # check doctr for model name
    # e.g., db_resnet50, fast_base
    def __init__(self, name='crnn_vgg16_bn', **kwargs) -> None:
        super().__init__(**kwargs)
        from doctr.models import recognition
        self.model = recognition.recognition_predictor(
            name,
            pretrained=True,
            symmetric_pad=True,
        )

    def to(self, device: str):
        self.model.to(device)

    def recognize(self, imgs: List[Image]) -> Result:
        imgs_tensor = torch.stack([img.to_channel_first().to_tensor() for img in imgs])
        texts, scores = list(zip(*self.model(imgs_tensor)))
        return Result(ocr_texts=texts, ocr_scores=scores)

class DoctrRecoDetector(BaseRecoDetector):
    def __init__(
            self,
            detector_name: str='fast_base',
            recognizer_name: str='crnn_vgg16_bn',
            assume_straight_pages=False,
            **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        
        from doctr.models import ocr_predictor
        args = {
            # 'preserve_aspect_ratio': False,
            # 'symmetric_pad': True,
            'assume_straight_pages': assume_straight_pages,
            'detect_language': True,
            'export_as_straight_boxes': False
        }
        self.model = ocr_predictor(
            det_arch=detector_name,
            reco_arch=recognizer_name,
            pretrained=True,
            **args
        )

    def to(self, device: str):
        self.model.to(device)

    def extract_from_block(self, block: Block):
        lines, line_bboxs, line_confidences, line_objectness_scores = [], [], [], []
        for line in block.lines:
            if len(line.words) == 0:                
                continue
            words, bbox, confidences, objectness_scores = [], None, [], []
            for word in line.words:
                words.append(word.value)
                confidences.append(word.confidence)
                objectness_scores.append(word.objectness_score)
                if bbox is None:
                    bbox = Bbox.from_np(np.array(word.geometry))
                else:
                    bbox = bbox.merge(Bbox.from_np(np.array(word.geometry)))
            lines.append(' '.join(words))
            line_bboxs.append(bbox)
            line_confidences.append(np.mean(confidences))
            line_objectness_scores.append(np.mean(objectness_scores))
        return lines, line_bboxs, line_confidences, line_objectness_scores

    def extract_from_page(self, page: Page):
        texts, bboxs, recognition_scores, detection_scores = [], [], [], []
        for block in page.blocks:
            _texts, _bboxs, _recognition_scores, _detection_scores = self.extract_from_block(block)
            texts += _texts
            bboxs += _bboxs
            recognition_scores += _recognition_scores
            detection_scores += _detection_scores
        return texts, bboxs, recognition_scores, detection_scores

    def recodetect(self, imgs: List[Image]) -> List[Result]:
        output = self.model([img.image for img in imgs])

        results = []
        for img, page in zip(imgs, output.pages):
            words, bboxs, recognition_scores, detection_scores = self.extract_from_page(page)
            assert(len(words) == len(bboxs) == len(recognition_scores) == len(detection_scores))
            bboxs = [bbox.scale(img.width, img.height) for bbox in bboxs]
            results.append(
                Result(
                    ocr_texts=words,
                    ocr_scores=recognition_scores,
                    bboxs=bboxs,
                    bbox_scores=detection_scores
                )
            )
        return results
