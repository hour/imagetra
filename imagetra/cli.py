from imagetra.api.base import BaseClient
from imagetra.api import flask, socket, websocket

CLIENT_MAP = {
    'flask': flask.FlaskClient,
    'socket': socket.SocketClient,
    'websocket': websocket.WebSocketClient,
}

def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="cli")
    parser.add_argument('--type', '-t', choices=list(CLIENT_MAP.keys()), default=list(CLIENT_MAP.keys())[0])
    BaseClient.add_argument(parser)
    args = parser.parse_args(argv)

    try:
        camid = int(args.camid)
    except:
        camid = args.camid
    CLIENT_MAP[args.type](host=args.hostname, port=args.port).run(camid=camid)

if __name__ == '__main__':
    main()