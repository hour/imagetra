# Dependency
- Python==3.11.0
- CUDA==12.6.3
- [uv](https://github.com/astral-sh/uv)

# Installation
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[all]
```

# Quick Start

```
imagetra_translate -c configs/nllb.yaml <input_image_or_video_path> <output_image_or_video_path>
```

# Python Code
```
from imagetra.detector.doctr import DoctrRecoDetector
from imagetra.translator.nllb import NLLBTranslator
from imagetra.editor.render import RenderEditor

# initialize each modules
recodetector = DoctrRecoDetector()
translator = NLLBTranslator(
    'facebook/nllb-200-distilled-600M',
    trg_lang='deu_Latn',
)
editor = RenderEditor() # add `font=<font path>` argument to support non-ASCII texts.

# build a pipeline
from imagetra.pipeline import Image2Image
pipeline = Image2Image(
    recodetector=ocr,
    translator=mt,
    editor=editor
)

# translate an image
from imagetra.common.media import Image
img = Image.load('data/image1.jpeg')
out = pipeline([img])
out.save(f'data/out_image.jpeg')

# translate a video
from imagetra.common.media import Video
video = Video.load("data/video.mp4")
out_frames = pipeline(video.frames)

for frame_index, frame in enumerate(out_frames):
    video.replace(frame, frame_index)

video.save("data/out_video.mp4")
```

## [Optional] [みんなの自動翻訳@TexTra API](https://mt-auto-minhon-mlt.ucri.jgn-x.jp/)
We also support "みんなの自動翻訳@TexTra" API for the translator module. To run it, we need [trans](https://github.com/ideuchi/trans) and configure as follows.

```
git clone https://github.com/ideuchi/trans

export HOME_TRANS=$(pwd)/trans
export TEXTRA_NAME=<your user_id>
export TEXTRA_KEY=<your api_key>
export TEXTRA_SECRET=<your api_secret>
```

- Visit [みんなの自動翻訳＠TexTra website](https://mt-auto-minhon-mlt.ucri.jgn-x.jp) to create a free account.
- After login, check [this](https://mt-auto-minhon-mlt.ucri.jgn-x.jp/content/api/) to get the user ID, API key, and API secret.
- Then, run `translate.py` using `configs/textra.yaml` as follows.

```
python translate.py -c configs/textra.yaml <input_image_or_video_path> <output_image_or_video_path>
```



# Citation
```
@article{kaing2024towards,
  title={Towards Scene Text Translation for Complex Writing Systems},
  author={Kaing, Hour and Song, Haiyue and Ding, Chenchen and Mao, Jiannan and Tanaka, Hideki and Utiyama, Masao},
  journal={言語処理学会 第30回年次大会},
  year={2025}
}
```