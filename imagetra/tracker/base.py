from imagetra.common.bbox import Bbox
from imagetra.common.media import Image
from typing import List, Tuple, Callable

class StrId:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.text2id = {}
        self.id2text = {}

    def add(self, text):
        if text in self.text2id:
            return False
        ind = len(self.text2id)
        assert(ind not in self.id2text)
        self.text2id[text] = ind
        self.id2text[ind] = text
        return True

    def id(self, text):
        return self.text2id[text]

    def text(self, ind):
        return self.id2text[ind]

    def add_multi(self, texts):
        count = 0
        for text in texts:
            if self.add(text):
                count += 1
        return count

    def ids(self, texts):
        return [self.id(text) for text in texts]

    def texts(self, ids):
        return [self.text(ind) for ind in ids]

class TrackerDataHolder:
    def __init__(self, fn_process) -> None:
        self.data = {}
        self.fn_process = fn_process
        self.clss_dict = StrId()

    def reset(self):
        self.data = {}

    def add(self, ind, bbox, score, cls, frame):
        return self.add_multi([ind], [bbox], [score], [cls], frame) > 0

    def add_multi(self, ids, bboxs, scores, clss, frame):
        uc_ids, uc_bboxs, uc_scores, uc_clss = [], [], [], []
        for ind, bbox, score, cls in zip(ids, bboxs, scores, clss):
            if ind in self.data and score < self.data[ind][0]:
                continue
            uc_ids.append(ind)
            uc_bboxs.append(bbox)
            uc_scores.append(score)
            uc_clss.append(cls)
        
        if len(uc_ids) < 0:
            return 0

        args = {
            'ids': uc_ids,
            'bboxs': uc_bboxs,
            'scores': uc_scores,
            'clss': uc_clss,
            'frame': frame,
            'clss_dict': self.clss_dict
        }

        for ind, score, bbox, entry in zip(uc_ids, uc_scores, uc_bboxs, self.fn_process(args)):
            self.data[ind] = (score, entry, bbox)

        return len(uc_ids)
    
    def get(self, ind):
        return self.data[ind][1]
    
    def get_multi(self, ids):
        return [self.data[ind][1] for ind in ids]
    
class BaseTracker:
    def __init__(
            self, 
            fn_process: Callable=None,
        ) -> None:
        self.dataholder = TrackerDataHolder(fn_process=fn_process) if fn_process is not None else None
        self.reset()
    
    def reset(self):
        raise NotImplementedError()

    def track(
            self,
            bboxs: List[Bbox],
            frame: Image, 
            scores: List[float]=None, 
            clss: List[float]=None
        ) -> Tuple[list, list, list, list]:
        raise NotImplementedError()
        
    def track_hold(self, bboxs: List[Bbox], frame: Image, scores: List[float]=None, clss: List[float]=None):
        assert(self.dataholder is not None)

        if scores is None:
            scores = [1.0] * len(bboxs)
        
        if clss is None:
            clss = [0.0] * len(bboxs)

        tracked_bboxs, tracked_ids, clss, tracked_scores = self.track(
            bboxs, frame, scores, clss
        )
        self.dataholder.add_multi(tracked_ids, tracked_bboxs, tracked_scores, clss, frame)
        return self.dataholder.get_multi(tracked_ids), tracked_bboxs, clss
    
    def track_hold_str(self, bboxs: List[Bbox], frame: Image, scores: List[float]=None, clss: List[str]=None):
        assert(self.dataholder.clss_dict is not None)
        self.dataholder.clss_dict.add_multi(clss)
        data, tracked_bboxs, clss = self.track_hold(bboxs, frame, scores, self.dataholder.clss_dict.ids(clss))
        return data, tracked_bboxs, self.dataholder.clss_dict.texts(clss)

