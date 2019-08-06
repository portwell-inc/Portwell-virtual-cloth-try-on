from flask import Flask, render_template,redirect,url_for,request,jsonify,flash,session
from flask_session import Session
from base64 import b64decode
from VITON.VITON import VITON, viton_model_init
from subprocess import run, PIPE
import subprocess
import json
import cv2
import os,sys

app = Flask(__name__)
#SESSION_TYPE = 'redis'
app.config.from_object(__name__)
app.secret_key = "025300a65059e046175068af08abe39d"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/demo")
def demo():
    return render_template('basic.html')

@app.route("/new_picture_api",methods=['POST','GET'])
def new_picture_api():
    if request.method == "GET":

        # get image uri and decode
        image = request.values.get("image")
        data_uri = image
        header, encoded = data_uri.split(",", 1)
        data = b64decode(encoded)
        with open("shot.jpg", "wb") as f:
            f.write(data)
    
        #run subprocess and load data to session
        run('python new_picture.py')
        image = cv2.imread('image.jpg')
        parse = cv2.imread('parse.jpg')
        with open("keypoint.json") as f:
            keypoint = json.load(f)
        session['keypoint'] = keypoint
        session['image'] = image
        session['parse'] = parse
        os.remove('image.jpg')
        os.remove('parse.jpg')
        os.remove('keypoint.json')
    
    return jsonify('OK')

# @app.route("/get")
# def get():
#     keypoint = session.get('keypoint', 'not set')
#     print(keypoint, file=sys.stderr)
#     return 'OK'

@app.route("/reset")
def reset():
    session.clear()
    return 'OK'

# @app.route("/VTO_api")
# def VTO_api():
#     stage1_model, stage2_model = viton_model_init()
#     result = VITON(c, cm_array, im, parse_array, pose_label, stage1_model, stage2_model)
#     return 


if __name__ == '__main__':
    app.debug = True
    app.run()
    sess = Session()
    sess.init_app(app)
    #print(, file=sys.stderr)