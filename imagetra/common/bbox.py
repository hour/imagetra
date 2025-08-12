from dataclasses import dataclass
import json, cv2
import numpy as np

@dataclass
class Coordinate:
    x: float
    y: float

    def to_tuple(self):
        return (float(self.x), float(self.y))

    def shift(self, x, y):
        return Coordinate(x=self.x+x, y=self.y+y)
    
    # [TODO]: remove, don't make sense.
    def scale(self, x, y):
        return Coordinate(x=self.x*x, y=self.y*y)

@dataclass
class Bbox:
    topleft: Coordinate
    topright: Coordinate
    bottomright: Coordinate
    bottomleft: Coordinate

    def to_list(self):
        return [
            self.topleft.to_tuple(),
            self.topright.to_tuple(),
            self.bottomright.to_tuple(),
            self.bottomleft.to_tuple(),
        ]

    def to_numpy(self):
        return np.array(self.to_list())

    def to_outbound_rectangle(self):
        min_x = self.min_x
        min_y = self.min_y
        return Bbox.from_wh(
            self.max_x - min_x,
            self.max_y - min_y,
            x=min_x, y=min_y
        )

    @classmethod
    def from_wh(cls, width, height, x=0, y=0):
        return cls(
            topleft=Coordinate(x, y),
            topright=Coordinate(x+width, y),
            bottomright=Coordinate(x+width, y+height),
            bottomleft=Coordinate(x, y+height),
        )

    @classmethod
    def from_np(cls, np_bbox):
        assert(np_bbox.shape == (4,2))
        return cls(
            topleft=Coordinate(*np_bbox[0]),
            topright=Coordinate(*np_bbox[1]),
            bottomright=Coordinate(*np_bbox[2]),
            bottomleft=Coordinate(*np_bbox[3]),
        )

    def arrange(self):
        coordinates = sorted(self.to_list(), key=lambda x: x[0])
        topleft, bottomleft = sorted(coordinates[:2], key=lambda x: x[1])
        topright, bottomright = sorted(coordinates[2:], key=lambda x: x[1])
        return Bbox(
            topleft=Coordinate(*topleft),
            topright=Coordinate(*topright),
            bottomright=Coordinate(*bottomright),
            bottomleft=Coordinate(*bottomleft),
        )
    
    def dumps(self):
        return json.dumps(self.to_list())

    def add_margin(self, margin):
        return Bbox(
            topleft=self.topleft.shift(-margin, -margin),
            topright=self.topright.shift(margin, -margin),
            bottomright=self.bottomright.shift(margin, margin),
            bottomleft=self.bottomleft.shift(-margin, margin)
        )
    
    def shift(self, x, y):
        return Bbox(
            topleft=self.topleft.shift(x, y),
            topright=self.topright.shift(x, y),
            bottomright=self.bottomright.shift(x, y),
            bottomleft=self.bottomleft.shift(x, y)
        )
    
    # [TODO]: remove, don't make sense.
    def scale(self, x, y):
        return Bbox(
            topleft=self.topleft.scale(x, y),
            topright=self.topright.scale(x, y),
            bottomright=self.bottomright.scale(x, y),
            bottomleft=self.bottomleft.scale(x, y)
        )

    def resize(self, nw, nh):
        min_x = self.min_x
        min_y = self.min_y
        rw = nw / (self.max_x - min_x)
        rh = nh / (self.max_y - min_y)

        def _scale_shift(coor):
            return Coordinate(
                x=coor.x * rw + min_x * (1 - rw),
                y=coor.y * rh + min_y * (1 - rh)
            )
        
        return Bbox(
            topleft=_scale_shift(self.topleft),
            topright=_scale_shift(self.topright),
            bottomright=_scale_shift(self.bottomright),
            bottomleft=_scale_shift(self.bottomleft)
        )

    @classmethod
    def loads(cls, json_dump):
        json_dump = json.loads(json_dump)
        return cls(
            topleft=Coordinate(*json_dump[0]),
            topright=Coordinate(*json_dump[1]),
            bottomright=Coordinate(*json_dump[2]),
            bottomleft=Coordinate(*json_dump[3]),
        )
    
    @staticmethod
    def save(path, bboxs):
        with open(path, 'w') as fout:
            for bbox in bboxs:
                print(bbox.dumps(), file=fout)

    @classmethod
    def load(cls, path):
        output = []
        with open(path, 'r') as fin:
            for line in fin:
                output.append(cls.loads(line))
        return output

    @property
    def min_x(self):
        xs, _ = list(zip(*self.to_list()))
        return min(xs)

    @property
    def min_y(self):
        _, ys = list(zip(*self.to_list()))
        return min(ys)

    @property
    def max_x(self):
        xs, _ = list(zip(*self.to_list()))
        return max(xs)

    @property
    def max_y(self):
        _, ys = list(zip(*self.to_list()))
        return max(ys)

    def area(self):
        return cv2.contourArea(self.to_numpy().astype(np.float32))

    def ap(self, gt_bbox):
        pts1 = self.to_numpy().astype(np.float32)
        pts2 = gt_bbox.to_numpy().astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(pts1, pts2)
        area1 = cv2.contourArea(pts1)
        return inter_area / area1 if area1 != 0 else 0.0
    
    def ar(self, gt_bbox):
        pts1 = self.to_numpy().astype(np.float32)
        pts2 = gt_bbox.to_numpy().astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(pts1, pts2)
        area2 = cv2.contourArea(pts2)
        return inter_area / area2 if area2 != 0 else 0.0

    def iou(self, gt_bbox):
        pts1 = self.to_numpy().astype(np.float32)
        pts2 = gt_bbox.to_numpy().astype(np.float32)
        area1 = cv2.contourArea(pts1)
        area2 = cv2.contourArea(pts2)
        inter_area, _ = cv2.intersectConvexConvex(pts1, pts2)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area != 0 else 0.0    

    def merge(self, bbox):
        topleft = Coordinate(
            x=min(self.topleft.x, bbox.topleft.x),
            y=min(self.topleft.y, bbox.topleft.y),
        )
        topright = Coordinate(
            x=max(self.topright.x, bbox.topright.x),
            y=min(self.topright.y, bbox.topright.y),
        )
        bottomright = Coordinate(
            x=max(self.bottomright.x, bbox.bottomright.x),
            y=max(self.bottomright.y, bbox.bottomright.y),
        )
        bottomleft = Coordinate(
            x=min(self.bottomleft.x, bbox.bottomleft.x),
            y=max(self.bottomleft.y, bbox.bottomleft.y),
        )
        return Bbox(
            topleft=topleft,
            topright=topright,
            bottomright=bottomright,
            bottomleft=bottomleft,
        )