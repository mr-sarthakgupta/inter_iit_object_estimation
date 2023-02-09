# -- coding: utf-8 --
import numpy as np
import cv2
from flask import Flask, request, json
import base64
import io
from controllers.dims_from_one_img import get_dims_from_one_img_1, get_dims_from_one_img_2, integrate_results

app = Flask(__name__)

@app.route("/scanner", methods=['POST'])
def get_volumetric_weight():
    uploaded_file = open('sample.json')
    uploaded_file = json.load(uploaded_file)
    x1 = base64.b64decode(uploaded_file['images'][0][1:])
    img_1 = open('decoded_1.png', 'wb')
    img_1.write(x1)
    x2 = base64.b64decode(uploaded_file['images'][1][1:])
    img_2 = open('decoded_2.png', 'wb')
    img_2.write(x2)
    return integrate_results(get_dims_from_one_img_1('decoded_1.png'), get_dims_from_one_img_2('decoded_2.png'))
