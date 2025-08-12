import cv2, lap
from typing import List, Tuple
import numpy as np

from imagetra.common.media import Image
from imagetra.common.bbox import Bbox
from imagetra.tracker.base import BaseTracker

DEFAULT_TRACKER_TYPE = 'CSRT'

TRACKER_NAMES = [
    'MIL',
    'GOTURN',
    'DaSiamRPN',
    'Nano',
    'Vit',
    'KCF',
    'CSRT',
    'BOOSTING',
    'TLD',
    'MEDIANFLOW',
    'MOSSE'
]

def get_tracker_by_name(name):
    if name == 'MIL' and hasattr(cv2, 'TrackerMIL_create'):
        return cv2.TrackerMIL_create()
    elif name == 'GOTURN' and hasattr(cv2, 'TrackerGOTURN_create'):
        return cv2.TrackerGOTURN_create()
    elif name == 'DaSiamRPN' and hasattr(cv2, 'TrackerDaSiamRPN_create'):
        return cv2.TrackerDaSiamRPN_create()
    elif name == 'Nano' and hasattr(cv2, 'TrackerNano_create'):
        return cv2.TrackerNano_create()
    elif name == 'Vit' and hasattr(cv2, 'TrackerVit_create'):
        return cv2.TrackerVit_create()
    elif name == 'KCF' and hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    elif name == 'CSRT' and hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif name == 'BOOSTING' and hasattr(cv2, 'TrackerBoosting_create'):
        return cv2.TrackerBoosting_create()
    elif name == 'TLD' and hasattr(cv2, 'TrackerTLD_create'):
        return cv2.TrackerTLD_create()
    elif name == 'MEDIANFLOW' and hasattr(cv2, 'TrackerMedianFlow_create'):
        return cv2.TrackerMedianFlow_create()
    elif name == 'MOSSE' and hasattr(cv2, 'TrackerMOSSE_create'):
        return cv2.TrackerMOSSE_create()
    else:
        raise ValueError(f"Tracker '{name}' is not available in this OpenCV version.")

def bbox2xywh(bbox):
    x = int(bbox.min_x)
    y = int(bbox.min_y)
    w = int(bbox.max_x - x)
    h = int(bbox.max_y - y)
    return x, y, w, h

def translate_bbox(xywh, bbox):
    shift_x = xywh[0] - bbox.min_x
    shift_y = xywh[1] - bbox.min_y
    return bbox.shift(shift_x, shift_y).resize(xywh[2], xywh[3])

class SingleCVTacker:
    def __init__(
            self,
            frame: Image,
            bbox: Bbox,
            score: float=1.0,
            cls: int=None,
            tracker_type=DEFAULT_TRACKER_TYPE,
            min_share_ratio: float=0.5,
            max_fail: int=5,
        ) -> None:
        self.tracker_type = tracker_type
        self.reset(frame, bbox, score, cls)
        self.min_share_ratio = min_share_ratio
        self.max_fail = max_fail

    def reset(self, frame: Image, bbox: Bbox, score: float=1.0, cls: int=None):
        self.tracker = get_tracker_by_name(self.tracker_type)
        self.tracker.init(frame.image.copy(), bbox2xywh(bbox))
        self.curr_bbox = bbox
        self.score = score
        self.cls = cls
        self.count_fail = 0

    def predict(self, frame):
        success, xywh = self.tracker.update(frame.image.copy())
        if success:
            self.curr_bbox = translate_bbox(xywh, self.curr_bbox)
        else:
            self.count_fail += 1
        if self.count_fail > self.max_fail:
            return None
        return self.curr_bbox

    def is_activate(self):
        return self.curr_bbox is not None

    def get_ious(self, bboxs: List[Bbox]):
        return [
            self.curr_bbox.iou(bbox)
            for bbox in bboxs
        ]

class CVTackers(BaseTracker):
    def __init__(self, tracker_type=DEFAULT_TRACKER_TYPE, **kwargs) -> None:
        self.trackers = []
        self.tracker_type = tracker_type
        super().__init__(**kwargs)
    
    def reset(self):
        self.trackers = []
    
    def track(
            self,
            bboxs: List[Bbox],
            frame: Image, 
            scores: List[float] = None, 
            clss: List[float] = None
        ) -> Tuple[list, list, list, list]:
        
        ious = []
        for tracker in self.trackers:
            tracker.predict(frame)
            ious.append(tracker.get_ious(bboxs))

        if len(ious) > 0:
            costs = 1 - np.array(ious)
            _, x, y = lap.lapjv(costs, extend_cost=True)
        else:
            x, y = [], [-1]*len(bboxs)

        for tracker_id, bbox_id in enumerate(x):
            assert(bbox_id >= -1)
            if bbox_id == -1 or min(costs[:, bbox_id]) == 1: # compare the score
                continue
            self.trackers[tracker_id].reset(frame, bboxs[bbox_id], scores[bbox_id], clss[bbox_id])
        
        for bbox_id, tracker_id in enumerate(y):
            if tracker_id != -1 and min(costs[:, bbox_id]) < 1:
                assert(tracker_id < len(self.trackers))
                continue # check this
            assert(tracker_id < 0 or min(costs[:, bbox_id]) == 1)
            self.trackers.append(
                SingleCVTacker(frame, bboxs[bbox_id], scores[bbox_id], clss[bbox_id])
            )

        tracked_bboxs, tracked_ids, tracked_clss, tracked_scores = [], [], [], []
        for tracker_id in range(len(self.trackers)):
            if not self.trackers[tracker_id].is_activate():
                continue
            tracked_ids.append(tracker_id)
            tracked_bboxs.append(self.trackers[tracker_id].curr_bbox)
            tracked_clss.append(self.trackers[tracker_id].cls)
            tracked_scores.append(self.trackers[tracker_id].score)

        return tracked_bboxs, tracked_ids, tracked_clss, tracked_scores
        