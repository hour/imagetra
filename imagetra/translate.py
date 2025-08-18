from imagetra.common.logger import get_logger, LogFile
from imagetra.common.filter import BboxFilter
from imagetra.tracker import boxmot
from imagetra import build_pipeline
from imagetra.common.metric import Timer
from imagetra.common.config import Config

import mimetypes, os
import numpy as np

homedir = os.path.dirname(__file__)
logger = get_logger('imagetra')

def run_video(args, pipeline, filter):
    from imagetra.common.media import Video
    video = Video.load(args.input)
    logger.info(f'Vidoe frames: {len(video)}')

    wkargs = {'tracker_type': args.tracker_type} if args.tracking else {}

    logfile = LogFile(args.logfile)

    timer = Timer().start()
    for i, result in enumerate(pipeline(video.frames, fn_filter=filter.filter, **wkargs)):
        if args.verbose:
            out_img = result.img.draw_bboxs(result.bboxs)
            out_img.image = np.hstack((out_img.image, result.img.image))
        else:
            out_img = result.img

        video.replace(out_img, i)
        if result.ocr_texts is not None and result.mt_texts is not None:
            for ocr_text, mt_text in zip(result.ocr_texts, result.mt_texts):
                logfile.print(f'frame@{i} : {ocr_text} -> {mt_text}')

        timer.track()
    
    logfile.print('-'*50)
    logfile.print(f'Execute time:\n{timer.format()}')
    logfile.print(timer.detail())
    logfile.close()
    
    video.save(args.output)

def run_imgs(args, pipeline, filter):
    from imagetra.common.media import Image
    img = Image.load(args.input)
    
    logfile = LogFile(args.logfile)

    timer = Timer().start()
    result = pipeline([img], fn_filter=filter.filter)[0]
    
    if args.verbose:
        out_img = result.img.draw_bboxs(result.bboxs)
        out_img.image = np.hstack((out_img.image, img.image))
    else:
        out_img = result.img

    out_img.save(args.output)

    if result.ocr_texts is not None and result.mt_texts is not None:
        for ocr_text, mt_text in zip(result.ocr_texts, result.mt_texts):
            logfile.print(f'{ocr_text} -> {mt_text}')

    timer.track()

    logfile.print('-'*50)
    logfile.print(f'Execute time:\n{timer.format()}')
    logfile.close()

def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--config', '-c', default=f'{homedir}/../configs/paddleocr.yaml')
    parser.add_argument('--logfile', '-l')
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--tracker_type', default=boxmot.DEFAULT_TRACKER_TYPE)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    filetype = mimetypes.guess_type(args.input)[0]
    config = Config.load_yaml(args.config)
    
    filter = BboxFilter(
        detect_min_score=config.recodetector_detect_min_score,
        recognize_min_score=config.recodetector_recognize_min_score,
        bbox_min_width=config.recodetector_min_width,
        bbox_min_height=config.recodetector_min_height,
    )

    pipeline_name = 'vdo2vdo' if args.tracking else 'img2img'
    pipeline = build_pipeline(pipeline_name, config, logfile=args.logfile)

    if config.common_greedy:
        pipeline = pipeline.iter

    if 'video' in filetype:
        run_video(args, pipeline, filter)
    elif 'image' in filetype:
        run_imgs(args, pipeline, filter)
    else:
        raise ValueError(f'Unknown type {filetype}')

if __name__ == '__main__':
    main()