from flask import Flask, render_template,redirect,url_for,request,jsonify,flash
from VITON.VITON import VITON, viton_model_init
from subprocess import run, PIPE
import subprocess
import cv2
import os
import time
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return return render_template('index.html')

@app.route("/new_picture_api")
def new_picture_api():
    ip = request.remote_addr
    input_dir = 'static/images/' + ip
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)
    # 寫入新圖片

    p = run('python new_picture.py',input=input_dir, encoding='ascii')
    return 'hello'

# @app.route("/VTO_api")
# def VTO_api():
#     stage1_model, stage2_model = viton_model_init()
#     result = VITON(c, cm_array, im, parse_array, pose_label, stage1_model, stage2_model)
#     return 


if __name__ == '__main__':
    app.debug = True
    app.run()
    #print(, file=sys.stderr)