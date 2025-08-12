from imagetra.editor.base import BaseEditor
from imagetra.common.media import Image, TextImage

from typing import List
from tqdm import tqdm

class RenderEditor(BaseEditor):
    def __init__(
            self,
            font: str = None,
            min_font_size: int=26,
            padding_lr: int=0,
            padding_tb: int=0,
            background='grey',
            fill: str='black',
            scale: float=1.0,
            num_workers: int=1,
            max_num_line: int=1,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.font = font
        self.min_font_size = min_font_size
        self.padding_lr = padding_lr
        self.padding_tb = padding_tb
        self.background = background
        self.fill = fill
        self.scale = scale
        self.num_workers = num_workers
        self.max_num_line = max_num_line

    def to(self, device: str):
        pass

    def render(self, text, width, height) -> Image:
        textimage = TextImage(
            text, font=self.font, width=width, height=height, 
            min_font_size=self.min_font_size,
            padding_lr=self.padding_lr,
            padding_tb=self.padding_tb,
            background=self.background,
            fill=self.fill,
            scale=self.scale,
            max_num_line=self.max_num_line,
        )
        return Image.from_pil(textimage.draw())

    # [TODO] parallel
    def edit(self, batch_texts: List[str], batch_imgs: List[Image]) -> Image:
        return [
            self.render(text, width=img.width, height=img.height)
            for img, text in zip(batch_imgs, batch_texts)
        ]
    
    def forward(self, texts: List[str], imgs: List[Image]) -> List[Image]:
        def _get_batch():
            pbar = (lambda x: tqdm(x, total=len(texts))) if self.show_pbar else (lambda x: x)
            out_texts, out_imgs = [], []
            for text, img in pbar(zip(texts, imgs)):
                if len(out_texts) > self.subbatch_size:
                    yield out_texts, out_imgs
                    out_texts, out_imgs = [], []
                out_texts.append(text)
                out_imgs.append(img)
            if len(out_texts) > 0:
                yield out_texts, out_imgs

        if self.num_workers < 2:
            return self.edit(texts, imgs)
        else:
            from multiprocessing import Pool
            pool = Pool(processes=self.num_workers)
            asynce_results = []

            for batch_texts, batch_imgs in _get_batch():
                asynce_results.append(
                    pool.apply_async(
                        self.edit, (batch_texts, batch_imgs)
                    )
                )
            
            pool.close()
            pool.join()

            output = []
            for result in asynce_results:
                output += result.get()
            return output


