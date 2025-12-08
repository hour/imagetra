from imagetra.common.logger import get_logger, LogFile
from imagetra.common.filter import BboxFilter
from imagetra.tracker import boxmot
from imagetra import build_pipeline
from imagetra.common.metric import Timer
from imagetra.common.config import Config

import mimetypes, os
import numpy as np
import cv2
from datetime import timedelta

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
    timer.track()
    
    if args.verbose:
        out_img = result.img.draw_bboxs(result.bboxs)
        out_img.image = np.hstack((out_img.image, img.image))
    else:
        out_img = result.img

    out_img.save(args.output)

    if result.ocr_texts is not None and result.mt_texts is not None:
        for ocr_text, mt_text in zip(result.ocr_texts, result.mt_texts):
            logfile.print(f'{ocr_text} -> {mt_text}')

    logfile.print('-'*50)
    logfile.print(f'Execute time:\n{timer.format()}')
    logfile.close()

def put_multiline_text(
    img,
    text,
    org=(10, 30),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    color=(0, 255, 0),
    thickness=2,
    line_gap=5
):
    x, y = org
    for i, line in enumerate(text.split("\n")):
        y_line = y + i * (int(font_scale * 30) + line_gap)
        cv2.putText(
            img,
            line,
            (x, y_line),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

def run_real_time(args, pipeline, filter):
    import cv2, time
    from imagetra.common.media import Image
    cap = cv2.VideoCapture(args.input)

    def _gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Error: Can't receive frame (stream end?). Exiting ...")
            frame = Image(frame, channel_first=False)
            yield frame
    
    # preprocess cap
    if args.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if args.scale is not None:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        new_width = int(width * args.scale)
        new_height = int(height * new_width / width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    wkargs = {'tracker_type': args.tracker_type} if args.tracking else {}

    timer = Timer().start()
    for result in pipeline(_gen(), fn_filter=filter.filter, **wkargs):
        timer.track()
        if args.verbose:
            out_img = result.img.draw_bboxs(result.bboxs)
            out_img.image = out_img.image.copy()
            curr_time = str(timedelta(seconds=timer.tracks[len(timer.tracks)-1]))
            put_multiline_text(
                out_img.image,
                f'Latency: {curr_time}\n'+timer.format(),
                org=(10, 30)
            )
        else:
            out_img = result.img
        
        cv2.imshow('Camera Feed', out_img.image)
        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input', help='Path to an image, a video, or a camera ID.')
    parser.add_argument('output', help='Path to the output. If the input is a camera ID, the output is ignored.')
    parser.add_argument('--config', '-c', default=f'{homedir}/../configs/paddleocr.yaml', help='Path to the configuration file.')
    parser.add_argument('--logfile', '-l', help='Path to the log file.')
    parser.add_argument('--tracking', action='store_true', help='Enable text tracking.')
    parser.add_argument('--tracker_type', default=boxmot.DEFAULT_TRACKER_TYPE, help='Type of tracker to use.')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--scale', type=float, help='Scaling factor for the camera frames.')
    parser.add_argument('--fps', type=int, help='Frames per second (FPS) of the camera.')
    args = parser.parse_args(argv)

    try:
        args.input = int(args.input)
        filetype = 'camera'
    except:
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

    if config.common_greedy or 'camera' in filetype:
        pipeline = pipeline.iter

    if 'video' in filetype:
        run_video(args, pipeline, filter)
    elif 'image' in filetype:
        run_imgs(args, pipeline, filter)
    elif 'camera' in filetype:
        run_real_time(args, pipeline, filter)
    else:
        raise ValueError(f'Unknown type {filetype}')

if __name__ == '__main__':
    main()