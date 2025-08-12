from dataclasses import dataclass
from imagetra.common.media import Image
from imagetra.common.bbox import Bbox

@dataclass
class Result:
    img: Image=None
    mt_texts: list[str]=None
    ocr_texts: list[str]=None
    ocr_scores: list[float]=None
    bboxs: list[Bbox]=None
    bbox_scores: list[float]=None
