from imagetra.common.base import Base
from imagetra.common.media import Image
from imagetra.common.logger import get_logger

from typing import List
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

logger = get_logger('Editor')

class ImagesManager:
    def __init__(self, min_score: float=0.6) -> None:
        self.key_imgs = []
        self.texts_imgs = []
        self.min_score = min_score

    def preprocess(self, img: Image):
        img = cv2.cvtColor(img.image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(img, (100, 100))

    def add(self, key_img: Image, key_text: str, value_img: Image):
        self.key_imgs.append(self.preprocess(key_img))
        self.texts_imgs.append({key_text: value_img})
    
    def add_text_img(self, des_id: int, key_text: str, value_img: Image):
        if key_text in self.texts_imgs[des_id]:
            return False
        self.texts_imgs[des_id][key_text] = value_img
        return True

    def get_scores(self, img: Image):
        scores = []
        img = self.preprocess(img)
        for des_id in range(len(self.key_imgs)):
            score = ssim(self.key_imgs[des_id], img)
            scores.append(score)
        return scores

    def query(self, img: Image, text:str):
        scores = self.get_scores(img)
        if len(scores) < 1:
            return None
        argmax_id = np.argmax(scores).item()
        if scores[argmax_id] < self.min_score:
            return None
        if text not in self.texts_imgs[argmax_id]:
            return argmax_id
        return self.texts_imgs[argmax_id][text]

class BaseEditor(Base):
    def __init__(self, subbatch_size=10, cache_img: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.subbatch_size = subbatch_size
        self.img_manager = ImagesManager() if cache_img else None

    def forward(self, texts: List[str], imgs: List[Image]) -> List[Image]:
        if self.img_manager is None:
            return self._edit(texts, imgs)
        else:
            return self._edit_cache(texts, imgs)

    def _edit(self, texts: List[str], imgs: List[Image]) -> List[Image]:
        length = len(texts)
        assert(length == len(imgs))
        def _get_batch():
            pbar = (lambda x: tqdm(x, total=length)) if self.show_pbar else (lambda x: x)
            for start in pbar(range(0, length, self.subbatch_size)):
                end = min(start+self.subbatch_size, length)
                yield texts[start:end], imgs[start:end]

        output = []
        for batch_texts, batch_imgs in _get_batch():
            output += self.edit(batch_texts, batch_imgs)
        return output

    def _edit_cache(self, texts: List[str], imgs: List[Image]) -> List[Image]:
        length = len(texts)
        assert(length == len(imgs))
        cached_ids = []
        cached_imgs = []

        def _get_batch():
            pbar = (lambda x: tqdm(x, total=length)) if self.show_pbar else (lambda x: x)
            out_texts, out_imgs, out_queries  = [], [], []
            for text_id in pbar(range(length)):
                img = imgs[text_id]
                text = texts[text_id]
                out_query = self.img_manager.query(img, text)

                if isinstance(out_query, Image):
                    assert(text_id not in cached_ids)
                    cached_ids.append(text_id)
                    cached_imgs.append(out_query)
                    continue

                if len(out_texts) > self.subbatch_size:
                    yield out_texts, out_imgs, out_queries
                    out_texts, out_imgs, out_queries = [], [], []

                out_texts.append(text)
                out_imgs.append(img)
                out_queries.append(out_query)

            if len(out_texts) > 0:
                yield out_texts, out_imgs, out_queries

        def _update_img_manager(_texts, _imgs, _queries, _out_edit):
            for i in range(len(_queries)):
                if _queries[i] is None:
                    self.img_manager.add(_imgs[i], _texts[i], _out_edit[i])
                elif isinstance(_queries[i], int):
                    self.img_manager.add_text_img(_queries[i], _texts[i], _out_edit[i])
                else:
                    raise NotImplementedError()

        output = []
        for batch_texts, batch_imgs, batch_queries in _get_batch():
            out_edit = self.edit(batch_texts, batch_imgs)
            _update_img_manager(batch_texts, batch_imgs, batch_queries, out_edit)
            output += out_edit
        
        final_output = [
            output.pop(0) if i not in cached_ids else cached_imgs[cached_ids.index(i)]
            for i in range(length)
        ]
        return final_output

    def edit(self, batch_texts: List[str], batch_imgs: List[Image]) -> List[Image]:
        raise NotImplementedError()

