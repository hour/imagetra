from imagetra.common.result import Result
from typing import List

class BboxFilter:
    def __init__(
            self,
            detect_min_score: int=0.9,
            recognize_min_score: int=0.9,
            bbox_min_width: int=64,
            bbox_min_height: int=64,
        ) -> None:
        self.detect_min_score = detect_min_score
        self.recognize_min_score = recognize_min_score
        self.bbox_min_width = bbox_min_width
        self.bbox_min_height = bbox_min_height
    
    def filter(self, results: List[Result]) -> List[Result]:
        def _keep(text, bbox, detection_score, recognition_score):
            if detection_score < self.detect_min_score:
                return False
            if recognition_score < self.recognize_min_score:
                return False
            if bbox.max_x - bbox.min_x < self.bbox_min_width:
                return False
            if bbox.max_y - bbox.min_y < self.bbox_min_height:
                return False
            return True

        for result in results:
            out_texts, out_bboxs, out_detection_scores, out_recognition_scores = [], [], [], []
            for i in range(len(result.ocr_texts)):
                if not _keep(
                    result.ocr_texts[i],
                    result.bboxs[i],
                    result.bbox_scores[i], 
                    result.ocr_scores[i]
                ):
                    continue
                out_texts.append(result.ocr_texts[i])
                out_bboxs.append(result.bboxs[i])
                out_detection_scores.append(result.bbox_scores[i])
                out_recognition_scores.append(result.ocr_scores[i])
            result.ocr_texts = out_texts
            result.bboxs = out_bboxs
            result.bbox_scores = out_detection_scores
            result.ocr_scores = out_recognition_scores
        return results