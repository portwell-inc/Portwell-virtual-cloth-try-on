from flask import Flask, render_template,redirect,url_for,request,jsonify,flash,session
from flask_session import Session
from VITON.VITON import VITON, viton_model_init
import torchvision.transforms as transforms
from subprocess import run
import json, os, sys
from PIL import Image
import numpy as np
import cv2
import base64

app = Flask(__name__)
app.config.from_object(__name__)
app.secret_key = "025300a65059e046175068af08abe39d"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/demo")
def demo():
    return render_template('demo.html')

@app.route("/tryon")
def tryon():
    di = 'static/images/cloth/'
    files_dir = os.listdir(di)
    cloth_dir = []
    for file in files_dir:
        new_file = '../' + di + file
        cloth_dir.append(new_file)
    return render_template('tryon.html',cloth_dir = cloth_dir)

@app.route("/new_picture_api",methods=['GET'])
def new_picture_api():
    if request.method == "GET":

        # get image uri and decode
        image = request.values.get("image")
        data_uri = image
        header, encoded = data_uri.split(",", 1)
        data = base64.b64decode(encoded)
        with open("shot.jpg", "wb") as f:
            f.write(data)
    
        #run subprocess and load data to session
        run('python new_picture.py')
        image = cv2.imread('image.jpg')
        parse = cv2.imread('parse.jpg', cv2.IMREAD_GRAYSCALE)
        keypoint = {}
        with open("keypoint.json") as f:
            keypoint = json.load(f)
        session['keypoint'] = keypoint
        session['image'] = image
        session['parse'] = parse
        os.remove('image.jpg')
        os.remove('parse.jpg')
        os.remove('keypoint.json')

    return jsonify('OK')

@app.route("/new_picture_get",methods=['GET'])
def new_picture_get():
    image = session.get('image', 'not set')

    if image != 'not set':
        image = cv2.resize(image, (300, 400), interpolation=cv2.INTER_CUBIC)
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        b64_code = str(base64.b64encode(img_str))
        b64_code = 'data:image/jpeg;base64,' + b64_code[2:-1]
        return jsonify({ 'image' : b64_code })
    else:
        return jsonify({ 'image' : 'not found' })

@app.route("/VTO_api",methods=['GET'])
def VTO_api():
    if request.method == "GET":
        #get cloht picture
        cloth_path = request.values.get("cloth")
        cloth_path = cloth_path[3:]
        cloth = cv2.imread(cloth_path)
        #get cloth mask
        mask_path = cloth_path.replace('cloth','cloth_mask')
        cloth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #get data from session
        image = session.get('image', 'not set')
        parse = session.get('parse', 'not set')
        keypoint = session.get('keypoint', 'not set')


        # cloth_mask = np.reshape(cloth_mask,(256,192))
        # parse = np.reshape(parse,(256,192))
        print(image.shape ,file=sys.stderr)
        print(cloth.shape ,file=sys.stderr)
        print(cloth_mask.shape ,file=sys.stderr)
        print(parse.shape ,file=sys.stderr)
        print(keypoint ,file=sys.stderr)

        #data transform
        transform = transforms.Compose([  \
            transforms.ToTensor(),   \
            transforms.Normalize([0.5], [0.5])])
        cloth = Image.fromarray(cv2.cvtColor(cloth,cv2.COLOR_BGR2RGB))
        cloth = transform(cloth)
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        image = transform(image)

        #call VITON api
        stage1_model, stage2_model = viton_model_init()
        result = VITON(cloth, cloth_mask, image, parse, keypoint, stage1_model, stage2_model)

    return jsonify('OK')

# @app.route("/get")
# def get():
#     keypoint = session.get('keypoint', 'not set')
#     image = session.get('image', 'not set')
#     parse = session.get('parse', 'not set')
#     return {'keypoint' : keypoint, 'image' : image, 'parse' : parse}

@app.route("/reset")
def reset():
    session.clear()
    return 'OK'

if __name__ == '__main__':
    app.debug = True
    app.run()
    sess = Session()
    sess.init_app(app)
    #print(, file=sys.stderr)