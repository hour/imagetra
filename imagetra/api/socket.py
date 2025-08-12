from imagetra.common.media import Image
from imagetra.api.base import BaseServer, BaseClient

import socket, cv2, time, struct
import numpy as np

BUFFER_SIZE = 65507  # Max size for a UDP packet

class SocketServer(BaseServer):
    def run(self, host: str = 'localhost', port: str = '0000'):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, int(port)))
        sock.listen(1)
        print("Server listening...")

        while True:
            conn, addr = sock.accept()
            print(f"Connected by {addr}")

            data = b""
            payload_size = struct.calcsize("L")

            while True:
                while len(data) < payload_size:
                    packet = conn.recv(4096)
                    if not packet: break
                    data += packet
                
                if len(data) < 1:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += conn.recv(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = Image(cv2.imdecode(frame, cv2.IMREAD_COLOR), channel_first=False)
                output, translations = self.translate(frame)
                output = cv2.imencode('.jpg', output.image)[1].tobytes()
                output = struct.pack("L", len(output)) + output
                conn.sendall(output)

class SocketClient(BaseClient):
    def __init__(self, host='localhost', port='0000') -> None:
        self.host = host
        self.port = port

    def run(self, camid=0, fn_update_cap=lambda x: x):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, int(self.port)))
            cap = cv2.VideoCapture(camid)
            fn_update_cap(cap)
            
            while True:
                local_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                msg = struct.pack("L", len(buffer)) + buffer.tobytes()
                sock.sendall(msg)
                
                # Receive and decode processed frame
                data = b""
                payload_size = struct.calcsize("L")
                while len(data) < payload_size:
                    data += sock.recv(4096)

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += sock.recv(4096)

                edited_frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = cv2.imdecode(np.frombuffer(edited_frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow("Processed Video", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                local_time = time.time() - local_time
                print(f'local time: {local_time}')

            cap.release()
            cv2.destroyAllWindows()
        