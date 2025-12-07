from imagetra.common.bbox import Bbox
from imagetra.common.logger import get_logger

import numpy as np
from dataclasses import dataclass
import torch
import cv2
from tqdm import tqdm
from typing import Self

from PIL import Image as PILImage
from PIL import ImageFont, ImageDraw, ImageColor

logger = get_logger('Media')

# library:
# - pytorchvideo

# references:
# - https://huggingface.co/docs/transformers/en/tasks/video_classification
# - https://pytorchvideo.org/docs/tutorial_torchhub_detection_inference
# - https://mindee.github.io/doctr/modules/models.html#

@dataclass
class Image:
    image: np.ndarray
    channel_first: bool = False

    @property
    def size(self):
        return self.image.shape

    @property
    def width(self):
        if self.channel_first:
            return self.size[2]
        return self.size[1]

    @property
    def height(self):
        if self.channel_first:
            return self.size[1]
        return self.size[0]
    
    @property
    def pil(self):
        return PILImage.fromarray(self.image.astype('uint8'), 'RGB')

    @classmethod
    def from_pil(cls, img_pil: PILImage):
        return cls(image=np.asarray(img_pil.convert('RGB')))

    def crop(self, left: float, top: float, right: float, bottom: float):
        return Image.from_pil(self.pil.crop((left, top, right, bottom)))

    def add_margin(self, left: float, top: float, right: float, bottom: float):
        return Image.from_pil(self.pil.crop((
            -left, -top, self.width + right, self.height + bottom
        )))

    def paste(self, image: Self, bbox: Bbox):
        img = self.pil
        img.paste(image.pil, (int(bbox.topleft.x), int(bbox.topleft.y)))
        return Image(
            image=np.array(img),
            channel_first=self.channel_first
        )

    def resize(self, width, height):
        if width == self.width and height == self.height:
            return self
        
        img = self.pil
        img = img.resize((width, height))
        return Image(
            image=np.asarray(img),
            channel_first=self.channel_first
        )

    def to_channel_first(self):
        if self.channel_first:
            return self
        return Image(
            image=np.stack((self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]), axis=0),
            channel_first=True
        )

    def to_channel_last(self):
        if not self.channel_first:
            return self
        return Image(
            image=np.stack((self.image[0], self.image[1], self.image[2]), axis=2),
            channel_first=False
        )

    def to_tensor(self, dtype=torch.uint8):
        return torch.from_numpy(self.image).type(dtype)
    
    def pad(self, right, bottom, left=0, top=0, mode='constant', constant_values=0):
        if self.channel_first:
            image = np.pad(self.image, ((0, 0), (top, bottom), (left, right)), mode=mode, constant_values=constant_values)
        else:
            image = np.pad(self.image, ((top, bottom), (left, right), (0, 0)), mode=mode, constant_values=constant_values)

        return Image(
            image=image,
            channel_first=self.channel_first
        )
    
    def save(self, path):
        self.to_channel_last().pil.save(path)

    @classmethod
    def load(cls, path):
        img = PILImage.open(path).convert('RGB')
        return cls(image=np.asarray(img))
    
    def copy(self):
        return Image(
            image=np.copy(self.image),
            channel_first=self.channel_first
        )

    def draw_bboxs(self, bboxs: list[Bbox], outline='red', width=4, texts: list[str]=None):
        font = ImageFont.load_default()
        img_pil = self.pil
        draw = ImageDraw.Draw(img_pil)
        if texts is None:
            texts = [None] * len(bboxs)

        for bbox, text in zip(bboxs, texts):
            draw.polygon(bbox.to_list(), outline=outline, width=width)
            if text is not None:
                draw.text(bbox.topleft.shift(5,5).to_tuple(), text, fill=outline, font=font)
        return Image.from_pil(img_pil)

class Video:
    def __init__(self, fframes, fourcc, fps):
        self.frames = fframes
        self.fourcc = fourcc
        self.fps = fps

    def __getitem__(self, frame_index) -> Image:
        return self.frames[frame_index]

    def __len__(self):
        return len(self.frames)

    @property
    def size(self):
        return self.frames[0].size

    @property
    def width(self):
        return self.frames[0].width

    @property
    def height(self):
        return self.frames[0].height

    # def __call__(self, frame_index) -> Image:
    #     return self.frames[frame_index]

    def replace(self, img: Image, frame_index):
        self.frames[frame_index] = img

    def save(self, path):
        video_writer = cv2.VideoWriter(path, 
            self.fourcc, self.fps, 
            (self.width, self.height)
        )

        for _, img in enumerate(self):
            video_writer.write(img.image.astype(np.uint8))

        video_writer.release()

    @classmethod
    def load(cls, path):
        vidcap = cv2.VideoCapture(path)
        # self.fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = []

        pbar = tqdm()
        while True:
            success, np_img = vidcap.read()
            if not success:
                break
            frames.append(Image(image=np_img))
            pbar.update(1)
        pbar.close()

        vidcap.release()
        return cls(frames, fourcc, fps)

    def draw_bboxs_at_frame(self, bboxs, frame_id):
        bboxs = [np.array(bbox.to_numpy(), dtype=np.int32).reshape((-1, 1, 2)) for bbox in bboxs]
        drawed_frame = cv2.polylines(self.frames[frame_id].image.copy(), bboxs, isClosed=True, color=(0, 0, 255), thickness=3)
        return Image(drawed_frame, channel_first=self.frames[frame_id].channel_first)

    def draw_bboxs(self, frame_bboxs):
        frames = []
        for frame_id in range(len(self.frames)):
            if frame_id < len(frame_bboxs):
                drawed_frame = self.draw_bboxs_at_frame(frame_bboxs[frame_id], frame_id)
            else:
                drawed_frame = self.frames[frame_id].copy()
            frames.append(drawed_frame)                
        return Video(frames, self.fourcc, self.fps)

class TextImage:
    def __init__(
        self,
        text: str, 
        font: str, 
        width: int, 
        height: int, 
        min_font_size: int, 
        padding_lr: float=0,
        padding_tb: float=0,
        background='grey',
        fill='black',
        scale: float=1.0,
        max_num_line: int=1,
    ):
        self.text = text
        self.width = width
        self.height = height
        self.scale = scale
        self.background = ImageColor.getrgb(background) if isinstance(background, str) else background
        self.fill = fill
        self.padding_lr = padding_lr
        self.padding_tb = padding_tb
        self.max_num_line = max_num_line
        self.font = self.init_font(font, min_font_size) # name or path

    def init_font(self, font, min_font_size):
        def _create_font(size):
            try:
                return ImageFont.truetype(f'{font}', size)
            except:
                return ImageFont.load_default(size=size)

        font_size, size = min_font_size, None
        offset_width, offset_height = self.padding_lr*2, self.padding_tb*2
        width, height = self.width - offset_width, self.height - offset_height
        draw = ImageDraw.Draw(PILImage.new("RGB", (width, height), self.background))

        def _width_height(font):
            left, top, right, bottom = draw.multiline_textbbox((0,0), self.text, font=font)
            _width = right + left
            _height = bottom + top
            return _width, _height
        
        font = _create_font(font_size)

        while (size is None or size[0] < width or size[1] < height) and font_size > 0:
            font = font.font_variant(size=font_size)
            size = _width_height(font)
            alpha = min(width-size[0], height-size[1]) // 10
            if alpha == 0:
                break
            font_size += alpha
            
        font_size = max(font_size, min_font_size)
        return font.font_variant(size=font_size)

    def draw(self) -> PILImage:
        img = PILImage.new('RGB', (self.width, self.height), self.background) # black background
        draw = ImageDraw.Draw(img)
        draw.text(
            (self.width//2, self.height//2), # left spacing and vertically middle
            self.text,
            fill=self.fill,
            font=self.font,
            anchor="mm",
        )
        
        assert(not (np.asarray(img)==self.background).all())
        return img
    