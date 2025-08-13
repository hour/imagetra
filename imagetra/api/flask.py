from imagetra.common.media import Image
from imagetra.api.base import BaseServer, BaseClient
from imagetra.common.logger import get_logger
from imagetra.tracker import boxmot

import base64, cv2
import numpy as np
import requests, time

def b64_to_image(b64_string):
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def image_to_b64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

class FlaskServer(BaseServer):
    """
    Example:
    FlaskServer(
        pipeline=pipeline, fn_filter=filter.filter
    ).run(
        host='localhost', port='8000'
    )
    """
    
    def run(self, host:str='localhost', port: str='8000', tracker_type=boxmot.DEFAULT_TRACKER_TYPE):
        logger = get_logger('FlaskServer')

        from flask import Flask, request, jsonify
        import time

        tracker = self.build_tracker(tracker_type=tracker_type)

        app = Flask(__name__)
        @app.route('/vdo2vdo', methods=['POST'])
        def handle_post():
            # recieved
            content = request.get_json()
            b64_string = content['image_base64']

            if not b64_string:
                return jsonify({'error': 'No image_base64 provided'}), 400

            img = Image(b64_to_image(b64_string), channel_first=content['channel_first'])
            img_id = content['image_id']
            logger.info(img_id)
            if img_id == 0:
                is_reset = True
                tracker.reset()
            else:
                is_reset = False

            # translation
            start = time.time()
            result = self.translate(img, tracker)
            end = time.time()

            # send back
            return jsonify({
                'status': 'success',
                'image_base64': image_to_b64(result.img.image),
                'channel_first': result.img.channel_first,
                'translations': result.mt_texts,
                'time': end - start,
                'is_reset': is_reset,
            })
        print(host, port)
        app.run(host=host, port=port)

class FlaskClient(BaseClient):
    """
    FlaskClient(
        host='localhost', port='8000'
    ).run(camid=0)
    """

    def __init__(self, host='localhost', port='8000') -> None:
        super().__init__(host, port)
        self.url = f'http://{host}:{port}/vdo2vdo'

    def format(self, img: Image, img_id: int=0):
        return {
            'image_base64': image_to_b64(img.image),
            'channel_first': img.channel_first,
            'image_id': img_id,
        }

    def translate(self, img: Image, img_id: int=0) -> Image:
        content = self.format(img, img_id)
        response = requests.post(self.url, json=content)
        if not response.ok:
            return None, None, None
        res_content = response.json()
        translations = res_content['translations']
        time =  res_content['time']
        output = b64_to_image(res_content['image_base64'])

        if res_content['is_reset']:
            print('is_reset')

        return Image(output, channel_first=res_content['channel_first']), translations, time

    def run(self, camid=0, fn_update_cap=lambda x: x):
        logger = get_logger('FlaskClient')

        cap = cv2.VideoCapture(camid)
        fn_update_cap(cap)

        frame_id = 0
        while True:
            # Capture frame-by-frame
            local_time = time.time()
            ret, frame = cap.read()

            # If frame read is not successful, break the loop
            if not ret:
                logger.info("Error: Can't receive frame (stream end?). Exiting ...")
                break

            frame = Image(frame, channel_first=False)
            out_img, _, server_time = self.translate(frame, frame_id)
            # server_time = 0
            cv2.imshow('Camera Feed', out_img.image)
            frame_id += 1
            
            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

            local_time = time.time() - local_time
            
            logger.info(f'Server time: {server_time}, local time: {local_time}, diff: {local_time - server_time}')

        cap.release()
        cv2.destroyAllWindows()