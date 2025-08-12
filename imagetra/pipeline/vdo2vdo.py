from typing import List
from tqdm import tqdm

from imagetra.detector.base import BaseRecoDetector
from imagetra.editor.base import BaseEditor
from imagetra.translator.base import BaseTranslator
from imagetra.common.media import Image
from imagetra.common.logger import get_logger
from imagetra.common.result import Result
from imagetra.pipeline.img2img import Image2Image
from imagetra.tracker import boxmot, cvtracker

logger = get_logger('Video2Video')

class Video2Video(Image2Image):
    def __init__(
            self, 
            recodetector: BaseRecoDetector=None,
            translator: BaseTranslator=None,
            editor: BaseEditor=None,
            min_share_ratio: float = 0.5,
            **kwargs,
        ) -> None:
        super().__init__(
            recodetector=recodetector,
            translator=translator,
            editor=editor,
            **kwargs
        )
        self.min_share_ratio = min_share_ratio
        
        recodetector.show_pbar = False
        editor.show_pbar = False
        translator.show_pbar = False

    def _translate_and_edit(self, texts, bboxs, main_img):
        translations = self._translate_texts(texts)
        out_imgs = self._transform([main_img], [bboxs])
        assert(len(out_imgs) == len(translations))
        out_imgs = self._edit(translations, out_imgs)
        return out_imgs, translations

    def _translate_with_tracker(self, bboxs, scores, clss, frame, tracker):
        # [TODO]: compare source image with tracked image to decide when to renew the cached.
        # - extract features using statistical methods such as color histograms, Texture analysis, and edge detection
        # - compare similarity of source and tracked image with threshold under `n`.

        data, tracked_bboxs, tracked_texts = tracker.track_hold_str(
            bboxs, frame, scores, clss
        )
        assert(len(data) == len(tracked_bboxs))

        trans_imgs, trans_texts = [], []
        for trans_img, trans_text in data:
            trans_imgs.append(trans_img)
            trans_texts.append(trans_text)
        return trans_imgs, trans_texts, tracked_bboxs, tracked_texts

    def build_tracker(self, tracker_type=cvtracker.DEFAULT_TRACKER_TYPE):
        def process_dataholder(args):
            clss_dict = args['clss_dict']
            texts = clss_dict.texts(args['clss'])
            trans_imgs, trans_texts = self._translate_and_edit(texts, args['bboxs'], args['frame'])
            return list(zip(trans_imgs, trans_texts))

        if tracker_type in cvtracker.TRACKER_NAMES:
            tracker = cvtracker.CVTackers(
                tracker_type=tracker_type,
                fn_process=process_dataholder,
            )
        elif tracker_type in boxmot.TRACKER_NAMES:
            tracker = boxmot.BoxMOTTracker(
                tracker_type=tracker_type,
                fn_process=process_dataholder,
            )
        else:
            raise NotImplementedError()

        return tracker

    def range(self, length):
        pbar = (lambda x: tqdm(x, total=length)) if self.show_pbar else (lambda x: x)
        return pbar(range(length))

    def images2images(self, imgs: List[Image], src_lang: str=None, trg_lang: str=None, fn_filter=None, tracker_type=cvtracker.DEFAULT_TRACKER_TYPE):
        tracker = self.build_tracker(tracker_type)

        batch_bboxs, batch_detection_scores, batch_texts, _ = self._recodetect(imgs, fn_filter)

        batch_out_imgs, batch_out_bboxs, batch_out_texts, batch_out_trans = [], [], [], []
        for img_id in self.range(len(imgs)):
            bboxs, scores, texts = batch_bboxs[img_id], batch_detection_scores[img_id], batch_texts[img_id]

            trans_imgs, trans_texts, tracked_bboxs, tracked_texts = self._translate_with_tracker(
                bboxs, scores, texts, imgs[img_id], tracker
            )

            batch_out_imgs.append(trans_imgs)
            batch_out_bboxs.append(tracked_bboxs)
            batch_out_texts.append(tracked_texts)
            batch_out_trans.append(trans_texts)

        return batch_out_imgs, batch_out_bboxs, batch_out_texts, batch_out_trans
    
    def iter(self, iterimgs, fn_filter=None, tracker_type=cvtracker.DEFAULT_TRACKER_TYPE):
        tracker = self.build_tracker(tracker_type)

        for img in tqdm(iterimgs):
            bboxs, scores, texts, _ = self._recodetect([img], fn_filter)
            bboxs, scores, texts = bboxs[0], scores[0], texts[0]

            trans_imgs, trans_texts, tracked_bboxs, tracked_texts = self._translate_with_tracker(
                bboxs, scores, texts, img, tracker
            )

            out_img = self._insert([img], [trans_imgs], [tracked_bboxs], pbar=lambda x: x)[0]

            yield Result(
                img=out_img, 
                mt_texts=trans_texts, 
                ocr_texts=tracked_texts,
                bboxs=tracked_bboxs,
            )
