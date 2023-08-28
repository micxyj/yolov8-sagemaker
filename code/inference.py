import torch
from ultralytics import YOLO
import io
from PIL import Image
import base64
import json
import boto3


def input_fn(request_body, request_content_type):
    print(f"=================input_fn=================\n{request_content_type}\n{request_body}")
    input_data = json.loads(request_body)
    return input_data


def model_fn(model_dir):
    print("=================model_fn=================")
    model = YOLO('/opt/ml/model/yolov8n.pt')
    model.to('cuda')
    print("=================model load complete=================")
    return model


def predict_fn(input_data, model):
    print("=================predict=================")
    init_imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in input_data['input_images']]
    results = model(init_imgs, device='cuda')
    print("=================predict done=================")
    return results

    
def output_fn(prediction, content_type):
    print("=================output process start=================")
    res = {'results': []}

    for r in prediction:
        item = {}
        item['objects'] = json.loads(r.tojson())

        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1])
        byteImgIO = io.BytesIO()
        im.save(byteImgIO, "WEBP")
        byteImgIO.seek(0)
        byteImg = byteImgIO.read()
        imgstr = base64.b64encode(byteImg).decode('ascii')
        item['image'] = imgstr

        res['results'].append(item)
    print("=================output process done=================")
    return json.dumps(res)
