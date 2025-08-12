from imagetra.common.media import Image
from imagetra.pipeline.vdo2vdo import Video2Video
from imagetra.tracker import cvtracker

class BaseServer(Video2Video):
    def __init__(self, tracker_type=cvtracker.DEFAULT_TRACKER_TYPE, fn_filter=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tracker = self.build_tracker(tracker_type)
        self.fn_filter = fn_filter

    def translate(self, img: Image):
        bboxs, scores, texts, _ = self._recodetect([img], self.fn_filter)
        bboxs, scores, texts = bboxs[0], scores[0], texts[0]

        trans_imgs, trans_texts, tracked_bboxs, tracked_texts = self._translate_with_tracker(
            bboxs, scores, texts, img, self.tracker
        )
        translations = [
            f'{text} -> {translation}'
            for text, translation in zip(tracked_texts, trans_texts)
        ]
        output = self._insert([img], [trans_imgs], [tracked_bboxs], pbar=lambda x: x)[0]
        return output, translations

    def run(self, host:str='localhost', port: str='0000'):
        raise NotImplementedError()

class BaseClient:
    def __init__(self, host='localhost', port='0000') -> None:
        self.host = host
        self.port = port

    def run(self, camid=0, fn_update_cap=lambda x: x):
        raise NotImplementedError()