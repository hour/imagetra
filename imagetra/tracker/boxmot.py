from boxmot import create_tracker, get_tracker_config
import numpy as np
from typing import List
from pathlib import Path

from imagetra.common.bbox import Bbox
from imagetra.common.media import Image
from imagetra.tracker.base import BaseTracker

DEFAULT_TRACKER_TYPE = 'bytetrack'

TRACKER_NAMES = [
    "strongsort",
    "ocsort",
    "bytetrack",
    "botsort",
    "deepocsort",
    # "hybridsort", # [TODO]: support this tracker. input of tracker.update(*) differs with others
    "boosttrack",
]

def bboxs2xyxys(bboxs: List[Bbox], scores: List[float], clss: List[float]):
    return np.array([
        [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, score, cls]
        for bbox, score, cls in zip(bboxs, scores, clss)
    ])

def translate_bbox(bbox, xyxy):
    shift_x = xyxy[0] - bbox.min_x
    shift_y = xyxy[1] - bbox.min_y
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    return bbox.shift(shift_x, shift_y).resize(w, h)

class BoxMOTTracker(BaseTracker):
    def __init__(
            self, 
            tracker_type: str=DEFAULT_TRACKER_TYPE,
            **kwargs
        ) -> None:
        self.tracker_type = tracker_type
        super().__init__(**kwargs)
    
    def reset(self):
        self.tracker = create_tracker(
            self.tracker_type,
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            tracker_config=get_tracker_config(self.tracker_type),
            device='cpu',
            half=False,
            per_class=False,
            evolve_param_dict=None
        )

    def postprocess_tracked_results(self, tracked_results, bboxs):
        if tracked_results.size == 0:
            return [], [], [], []

        bbox_ids = tracked_results[:, 7].astype(np.int64).tolist()
        bboxs = [
            translate_bbox(bboxs[ind], xyxy)
            for ind, xyxy in zip(bbox_ids, tracked_results[:, :4])
        ]
        ids = tracked_results[:, 4].astype(np.int64).tolist()
        scores = tracked_results[:, 5].tolist()
        clss = tracked_results[:, 6].astype(np.int64).tolist()
        return ids, bboxs, clss, scores

    def track(self, bboxs: List[Bbox], frame: Image, scores: List[float]=None, clss: List[float]=None):
        if scores is None:
            scores = [1.0] * len(bboxs)
        
        if clss is None:
            clss = [0.0] * len(bboxs)

        tracked_results = self.tracker.update(
            bboxs2xyxys(bboxs, scores, clss),
            frame.image
        )

        tracked_ids, tracked_bboxs, clss, tracked_scores = self.postprocess_tracked_results(tracked_results, bboxs)
        
        if hasattr(self.tracker, 'active_tracks'):
            for track in self.tracker.active_tracks:
                if not hasattr(track, 'is_activated') or track.is_activated:
                    continue
                assert(track.conf >= self.tracker.det_thresh)
                det_ind = int(track.det_ind)
                tracked_ids.append(track.id)
                tracked_bboxs.append(translate_bbox(bboxs[det_ind], track.xyxy))
                clss.append(int(track.cls))
                tracked_scores.append(track.conf)

        # if hasattr(self.tracker, 'lost_stracks'):
        #     for track in self.tracker.lost_stracks:
        #         assert(track.conf >= self.tracker.det_thresh)
        #         tracked_ids.append(track.id)
        #         tracked_bboxs.append(translate_bbox(self.dataholder.data[track.id][2], track.xyxy))
        #         clss.append(int(track.cls))
        #         tracked_scores.append(track.conf)

        return tracked_bboxs, tracked_ids, clss, tracked_scores
