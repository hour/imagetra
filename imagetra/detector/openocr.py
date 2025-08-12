from imagetra.common.logger import get_logger
logger = get_logger('OpenOCRRecoDetector')

try:
    from openocr import OpenOCR
except ModuleNotFoundError as e:
    logger.error("The 'openocr' package is not installed. Install with `pip install imagetra[openocr]`")
    raise

import numpy as np
from typing import List

from imagetra.common.media import Image
from imagetra.common.bbox import Bbox
from imagetra.detector.base import BaseRecoDetector
from imagetra.common.result import Result

def format(result):
    texts, points, scores = [], [], []
    for ele in result:
        texts.append(ele['transcription'])
        points.append(Bbox.from_np(np.array(ele['points'])).arrange())
        scores.append(ele['score'])
    return Result(ocr_texts=texts, ocr_scores=scores, bboxs=points, bbox_scores=scores)

class OpenOCRRecoDetector(BaseRecoDetector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = OpenOCR(backend='onnx', device='cpu')

    def to(self, device: str):
        self.model = OpenOCR(backend='onnx', device=device)
        return self

    def recodetect(self, imgs: List[Image]) -> List[Result]:
        results, _ = self.model(
            img_numpy=[img.image for img in imgs]
        )
        return [
            format(result)
            for result in results
        ]
    