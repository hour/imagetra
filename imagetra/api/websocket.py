from imagetra.common.media import Image
from imagetra.api.base import BaseServer, BaseClient

import base64, cv2, time
import numpy as np
import asyncio, websockets

class WebSocketServer(BaseServer):
    def run(self, host: str = 'localhost', port: str = '0000'):
        async def process_frame(data: bytes) -> bytes:
            nparr = np.frombuffer(base64.b64decode(data), np.uint8)
            frame = Image(cv2.imdecode(nparr, cv2.IMREAD_COLOR), channel_first=False)
            output, translations = self.translate(frame)
            _, buffer = cv2.imencode('.jpg', output.image)
            return base64.b64encode(buffer).decode('utf-8')

        async def handler(websocket):
            print("Client connected")
            try:
                async for message in websocket:
                    edited_frame = await process_frame(message.encode('utf-8'))
                    await websocket.send(edited_frame)
            except websockets.ConnectionClosed:
                print("Client disconnected")

        async def _run():
            async with websockets.serve(handler, host, port):
                print("Server started")
                await asyncio.Future()  # Run forever

        asyncio.run(_run())

class WebSocketClient(BaseClient):
    def __init__(self, host='localhost', port='0000') -> None:
        super().__init__(host, port)
        self.url = f"ws://{host}:{port}"

    def run(self, camid=0, fn_update_cap=lambda x: x):
        async def send_video():
            async with websockets.connect(self.url) as websocket:
                cap = cv2.VideoCapture(camid)
                fn_update_cap(cap)

                while True:
                    local_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    _, buffer = cv2.imencode('.jpg', frame)
                    await websocket.send(base64.b64encode(buffer).decode('utf-8'))

                    # Receive and decode processed frame
                    response = await websocket.recv()
                    img = np.frombuffer(base64.b64decode(response), np.uint8)
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    cv2.imshow("Processed Video", img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    local_time = time.time() - local_time
                    print(f'local time: {local_time}')

                cap.release()
                cv2.destroyAllWindows()

        asyncio.run(send_video())