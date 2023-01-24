# -*- coding: utf-8 -*-
from flask import Flask, jsonify
from controllers.dims_from_one_img import get_dims_from_one_img

app = Flask(__name__)


@app.route("/scanner/<string:img_path>", methods=['GET', 'POST'])
def get_volumetric_weight(img_path):
    return get_dims_from_one_img(img_path)
