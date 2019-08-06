from flask import Flask, render_template,redirect,url_for,request,jsonify,flash,session
from flask_session import Session
import base64
from VITON.VITON import VITON, viton_model_init
from subprocess import run
import json, os, sys
import cv2

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
    return render_template('tryon.html')

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


# @app.route("/VTO_api")
# def VTO_api():

#     #get data from session
#     image = session.get('image', 'not set')
#     parse = session.get('parse', 'not set')
#     keypoint = session.get('keypoint', 'not set')

#     #get cloth mask

#     stage1_model, stage2_model = viton_model_init()
#     result = VITON(c, cm_array, im, parse_array, pose_label, stage1_model, stage2_model)
#     return 

# @app.route("/get")
# def get():
#     keypoint = session.get('keypoint', 'not set')
#     print(keypoint, file=sys.stderr)
#     return 'OK'

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