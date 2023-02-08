# -*- coding: utf-8 -*-
import numpy as np
import cv2
from flask import Flask, request
from controllers.dims_from_one_img import get_dims_from_one_img_1, get_dims_from_one_img_2, integrate_results

app = Flask(__name__)


@app.route("/scanner", methods=['POST'])
def get_volumetric_weight():
    nparr1 = np.fromstring(request.data["images"][0], np.uint8)
    nparr2 = np.fromstring(request.data["images"][1], np.uint8)
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    return integrate_results(get_dims_from_one_img_1(img1), get_dims_from_one_img_2(img2))
