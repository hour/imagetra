from imagetra.api.base import BaseClient
from imagetra.api import flask, socket, websocket
import cv2

CLIENT_MAP = {
    'flask': flask.FlaskClient,
    'socket': socket.SocketClient,
    'websocket': websocket.WebSocketClient,
}

def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="cli")
    parser.add_argument('--type', '-t', choices=list(CLIENT_MAP.keys()), default=list(CLIENT_MAP.keys())[0])
    parser.add_argument('--scale', type=float)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--verbose', '-v', action='store_true')
    BaseClient.add_argument(parser)
    args = parser.parse_args(argv)

    try:
        camid = int(args.camid)
    except:
        camid = args.camid

    def fn_update_cap(cap):
        if args.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        
        if args.scale is not None:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            new_width = int(width * args.scale)
            new_height = int(height * new_width / width)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    CLIENT_MAP[args.type](host=args.hostname, port=args.port).run(
        camid=camid, fn_update_cap=fn_update_cap, verbose=args.verbose
    )

if __name__ == '__main__':
    main()