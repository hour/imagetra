from imagetra.pipeline.vdo2vdo import Video2Video
from imagetra.pipeline.img2img import Image2Image
from imagetra.tracker import boxmot

from typing import Union
import os

homedir = os.path.dirname(__file__)

class BaseServer:
    def __init__(self, pipeline: Union[Video2Video, Image2Image], fn_filter=None) -> None:
        self.pipeline = pipeline
        if isinstance(pipeline, Video2Video):
            self.translate = lambda img, tracker: self.pipeline.image2image(img, tracker, fn_filter=fn_filter)
            self.build_tracker = lambda tracker_type=boxmot.DEFAULT_TRACKER_TYPE: self.pipeline.build_tracker(tracker_type)
        else:
            self.translate = lambda img, tracker=None: self.pipeline.image2image(img, fn_filter=fn_filter)
            self.build_tracker = lambda tracker_type=None: None

    def add_argument(parser):
        parser.add_argument('--hostname', '-n', default='localhost')
        parser.add_argument('--port', '-p', default='8000')
        parser.add_argument('--config', '-c', default=f'{homedir}/../../configs/doctr.yaml')
        parser.add_argument('--tracking', action='store_true')
        parser.add_argument('--tracker_type', default=boxmot.DEFAULT_TRACKER_TYPE)

    def run(self, host:str='localhost', port: str='8000'):
        raise NotImplementedError()

class BaseClient:
    def __init__(self, host='localhost', port='8000') -> None:
        self.host = host
        self.port = port

    def add_argument(parser):
        parser.add_argument('--hostname', '-n', default='localhost')
        parser.add_argument('--port', '-p', default='8000')
        parser.add_argument('--camid', default='0')

    def run(self, camid=0, fn_update_cap=lambda x: x):
        raise NotImplementedError()