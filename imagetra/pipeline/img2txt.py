from typing import List, Any

from imagetra.common.base import Base
from imagetra.detector.base import BaseRecoDetector
from imagetra.translator.base import BaseTranslator
from imagetra.common.media import Image
from imagetra.common.batch import flatten, unflatten
from imagetra.common.logger import get_logger
from imagetra.common.result import Result

logger = get_logger('Image2Text')

class Image2Text(Base):
    def __init__(
        self,
        recodetector: BaseRecoDetector=None,
        translator: BaseTranslator=None,
        logfile: str=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.recodetector = recodetector
        self.translator = translator
        self.logfile = logfile

    def _recodetect(self, imgs: List[Image], fn_filter=None):
        results = self.recodetector(imgs)
        if fn_filter is not None:
            results = fn_filter(results)
        batch_bboxs, batch_detection_scores, batch_texts, batch_recognition_scores = [], [], [], []
        for result in results:
            batch_bboxs.append(result.bboxs)
            batch_detection_scores.append(result.bbox_scores)
            batch_texts.append(result.ocr_texts)
            batch_recognition_scores.append(result.ocr_scores)
        return batch_bboxs, batch_detection_scores, batch_texts, batch_recognition_scores

    def _translate_texts(self, texts, src_lang: str=None, trg_lang: str=None):
        translations = self.translator(texts, src_lang, trg_lang)
        return translations

    def forward(self, imgs: List[Image], src_lang: str=None, trg_lang: str=None, fn_filter=None, contextualize: bool=False) -> Any:
        batch_bboxs, _, batch_texts, _ = self._recodetect(imgs, fn_filter)
        
        if contextualize:
            ocr_texts = [self.translator.contextualize(texts) for texts in batch_texts]
        else:
            ocr_texts = flatten(batch_texts)
       
        batch_translations = self._translate_texts(ocr_texts, src_lang, trg_lang)

        if contextualize:
            batch_translations = [self.translator.decontextualize(translation) for translation in batch_translations]
        else:
            batch_translations = unflatten(batch_translations, batch_texts)

        assert(len(batch_translations) == len(batch_texts))

        return [
            Result(
                mt_texts=translations, 
                ocr_texts=texts,
                bboxs=bboxs
            )
            for translations, texts, bboxs in zip(batch_translations, batch_texts, batch_bboxs)
        ]
