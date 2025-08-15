from imagetra.api.base import BaseServer
from imagetra.api import flask, socket, websocket

SERVER_MAP = {
    'flask': flask.FlaskServer,
    'socket': socket.SocketServer,
    'websocket': websocket.WebSocketServer,
}

def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="serve")
    parser.add_argument('--type', '-t', choices=list(SERVER_MAP.keys()), default=list(SERVER_MAP.keys())[0])
    BaseServer.add_argument(parser)
    args = parser.parse_args(argv)

    from imagetra import build_pipeline
    from imagetra.common.config import Config
    from imagetra.common.filter import BboxFilter

    config = Config.load_yaml(args.config)
    pipeline_type = 'vdo2vdo' if args.tracking else 'img2img'
    pipeline = build_pipeline(pipeline_type, config)
    
    filter = BboxFilter(
        detect_min_score=config.recodetector_detect_min_score,
        recognize_min_score=config.recodetector_recognize_min_score,
        bbox_min_width=config.recodetector_min_width,
        bbox_min_height=config.recodetector_min_height,
    )

    SERVER_MAP[args.type](
        pipeline=pipeline, fn_filter=filter.filter
    ).run(host=args.hostname, port=args.port, tracker_type=args.tracker_type)

if __name__ == '__main__':
    main()