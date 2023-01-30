# -*- coding: utf-8 -*-
import numpy as np
import cv2
from flask import Flask, request
from controllers.dims_from_one_img import get_dims_from_one_img

app = Flask(__name__)


@app.route("/scanner", methods=['POST'])
def get_volumetric_weight():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return get_dims_from_one_img(img)
