from typing import List, Any
import numpy as np
from tqdm import tqdm

from imagetra.common.base import Base
from imagetra.detector.base import BaseRecoDetector
from imagetra.editor.base import BaseEditor
from imagetra.translator.base import BaseTranslator
from imagetra.common.media import Image
from imagetra.common.batch import flatten, unflatten
from imagetra.common.transform import transform
from imagetra.common.insert import insert
from imagetra.common.bbox import Bbox
from imagetra.common.logger import get_logger
from imagetra.common.result import Result

logger = get_logger('Image2Image')

class Image2Image(Base):
    def __init__(
        self,
        recodetector: BaseRecoDetector=None,
        translator: BaseTranslator=None,
        editor: BaseEditor=None,
        margin: float=0,
        logfile: str=None,
        fix_bbox: bool=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.recodetector = recodetector
        self.editor = editor
        self.translator = translator
        self.margin = margin
        self.fix_bbox = Bbox.from_wh(width=900, height=300) if fix_bbox else None
        self.logfile = logfile

        recodetector.show_pbar = False
        editor.show_pbar = False
        translator.show_pbar = False

    def _edit(self, translations: List[str], imgs: List[Image]):
        if len(translations) < 1:
            return []
        translations = np.array(translations)
        imgs = np.array(imgs)
        mask = np.char.strip(translations) != ''
        edited_imgs = np.full(translations.shape, None, dtype=object)
        edited_imgs[mask] = self.editor(
            translations[mask],
            imgs[mask]
        )
        # logger.info('Done edition')
        return edited_imgs

    def _recodetect(self, imgs: List[Image], fn_filter=None):
        results = self.recodetector(imgs)
        if fn_filter is not None:
            results = fn_filter(results)
        # logger.info('Done detection')
        batch_bboxs, batch_detection_scores, batch_texts, batch_recognition_scores = [], [], [], []
        for result in results:
            batch_bboxs.append(result.bboxs)
            batch_detection_scores.append(result.bbox_scores)
            batch_texts.append(result.ocr_texts)
            batch_recognition_scores.append(result.ocr_scores)
        return batch_bboxs, batch_detection_scores, batch_texts, batch_recognition_scores

    def _translate_texts(self, texts, src_lang: str=None, trg_lang: str=None):
        translations = self.translator(texts, src_lang, trg_lang)
        # logger.info('Done translation')
        return translations

    # [TODO] parallel insert
    def _insert(self, imgs, batch_edited_imgs, batch_bboxs, pbar=tqdm):
        output = []
        for img, bboxs, edited_imgs in pbar(zip(imgs, batch_bboxs, batch_edited_imgs)):
            for bbox, edited_img in zip(bboxs, edited_imgs):
                if edited_img is None:
                    continue
                img = insert(img, edited_img, bbox, margin=self.margin, copy=True)
            output.append(img)
        # logger.info('Done insertion')
        return output

    def _transform(self, imgs, batch_bboxs):
        flat_transformed_imgs = []
        for bboxs, img in zip(batch_bboxs, imgs):
            for bbox in bboxs:
                _bbox = bbox.add_margin(self.margin)
                if self.fix_bbox is not None:
                    _img = transform(img, _bbox, self.fix_bbox)
                else:
                    _img = transform(img, _bbox, _bbox.to_outbound_rectangle())
                flat_transformed_imgs.append(_img)
        return flat_transformed_imgs

    def images2images(self, imgs: List[Image], src_lang: str=None, trg_lang: str=None, fn_filter=None):
        batch_bboxs, _, batch_texts, _ = self._recodetect(imgs, fn_filter)
        flat_translations = self._translate_texts(flatten(batch_texts), src_lang, trg_lang)
        flat_transformed_imgs = self._transform(imgs, batch_bboxs)
        flat_edited_imgs = self._edit(flat_translations, flat_transformed_imgs)
        batch_edited_imgs = unflatten(flat_edited_imgs, batch_texts)
        batch_translations = unflatten(flat_translations, batch_texts)
        return batch_edited_imgs, batch_bboxs, batch_texts, batch_translations

    def forward(self, imgs: List[Image], src_lang: str=None, trg_lang: str=None, **kwargs) -> Any:
        assert((self.recodetector, self.translator, self.editor) != (None, None, None))
        batch_edited_imgs, batch_bboxs, batch_texts, batch_translations = self.images2images(imgs, src_lang, trg_lang, **kwargs)
        out_imgs = self._insert(imgs, batch_edited_imgs, batch_bboxs)
        
        return [
            Result(
                img=out_img, 
                mt_texts=translations,
                ocr_texts=texts,
                bboxs=bboxs,
            )
            for out_img, texts, bboxs, translations in zip(out_imgs, batch_texts, batch_bboxs, batch_translations)
        ]

    def iter(self, iterimgs, src_lang: str=None, trg_lang: str=None, fn_filter=None):
        for img in iterimgs:
            bboxs, scores, texts, _ = self._recodetect([img], fn_filter)
            bboxs, scores, texts = bboxs[0], scores[0], texts[0]

            trans_texts = self._translate_texts(texts, src_lang, trg_lang)
            transformed_imgs = self._transform([img], [bboxs])
            trans_imgs = self._edit(trans_texts, transformed_imgs)

            out_img = self._insert([img], [trans_imgs], [bboxs], pbar=lambda x: x)[0]

            yield Result(
                img=out_img, 
                mt_texts=trans_texts, 
                ocr_texts=texts,
                bboxs=bboxs,
            )

