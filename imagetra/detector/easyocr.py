from imagetra.common.logger import get_logger
logger = get_logger('EasyOCRRecoDetector')

try:
    import easyocr
except ModuleNotFoundError as e:
    logger.error("The 'easyocr' package is not installed. Install with `pip install imagetra[easyocr]`")
    raise

import requests 
requests.packages.urllib3.disable_warnings() 
import ssl 
try:
    _create_unverified_https_context = ssl._create_unverified_context 
except AttributeError: 
    pass 
else: 
    ssl._create_default_https_context = _create_unverified_https_context

import numpy as np
from typing import List

from imagetra.common.media import Image
from imagetra.common.bbox import Bbox
from imagetra.detector.base import BaseRecoDetector
from imagetra.common.result import Result

def format(result):
    texts, points, scores = [], [], []
    for point, text, score in result:
        texts.append(text)
        points.append(Bbox.from_np(np.array(point)).arrange())
        scores.append(score.item())
    return Result(ocr_texts=texts, ocr_scores=scores, bboxs=points, bbox_scores=scores)

class EasyOCRRecoDetector(BaseRecoDetector):
    def __init__(
            self, 
            langs: list=['en'],
            **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.model = easyocr.Reader(langs)
        self.langs = langs

    def to(self, device: str):
        self.model = easyocr.Reader(
            self.langs,
            gpu=True if device == 'cuda' else False
        )
        return self

    def recodetect(self, imgs: List[Image]) -> List[Result]:
        results = [
            self.model.readtext(img.image, workers=self.workers)
            for img in imgs
        ]
        return [format(result) for result in results]
    