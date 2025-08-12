from imagetra.common.logger import get_logger
logger = get_logger('PaddleOCRRecoDetector')

try:
    from paddleocr import PaddleOCR
except ModuleNotFoundError as e:
    logger.error("The 'paddleocr' package is not installed. Install with `pip install imagetra[paddleocr]`")
    raise

import numpy as np
from typing import List

from imagetra.common.media import Image
from imagetra.common.bbox import Bbox
from imagetra.detector.base import BaseRecoDetector
from imagetra.common.result import Result

def format(result):
    texts, points, scores = [], [], []
    if result is not None:
        for ele in result:
            texts.append(ele[1][0])
            points.append(Bbox.from_np(np.array(ele[0])).arrange())
            scores.append(ele[1][1])
    return Result(ocr_texts=texts, ocr_scores=scores, bboxs=points, bbox_scores=scores)

class PaddleOCRRecoDetector(BaseRecoDetector):
    def __init__(self, lang: str='en', **kwargs) -> None:
        super().__init__(**kwargs)
        self.lang = lang
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,
            show_log=False,
        )

    def to(self, device: str):
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang,
            use_gpu=True if device == 'cuda' else False,
            show_log=False,
        )
        return self

    def recodetect(self, imgs: List[Image]) -> List[Result]:
        results = [
            self.model.ocr(img.image, cls=True)
            for img in imgs
        ]
        return [format(result[0]) for result in results]
    