from flask import Flask, render_template,redirect,url_for,request,jsonify,flash,session
from multiprocessing import Pool
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
app.config['SESSION_FILE_DIR'] ='flask_session'
app.config['SESSION_FILE_THRESHOLD'] = 500
app.config['SESSION_USE_SIGNER'] = False
app.config['SESSION_PERMANENT'] = True
Session(app)

def Call_VITON(data):
    #get cloht picture
    cloth = cv2.imread(data[0])
    #get cloth mask
    mask_path = data[0].replace('cloth','cloth_mask')
    cloth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    #data transform
    transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize([0.5], [0.5])])
    cloth = Image.fromarray(cv2.cvtColor(cloth,cv2.COLOR_BGR2RGB))
    cloth = transform(cloth)
    image = Image.fromarray(cv2.cvtColor(data[1],cv2.COLOR_BGR2RGB))
    image = transform(image)

    #call VITON api
    stage1_model, stage2_model = viton_model_init()
    result = VITON(cloth, cloth_mask, image, data[2], data[3], stage1_model, stage2_model)

    return result


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

        session.clear()

        # get image uri and decode to numpy array
        image = request.values.get("image")
        data_uri = image
        header, encoded = data_uri.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.fromstring(data,np.uint8)
        image = cv2.imdecode(nparr,cv2.COLOR_BGR2RGB)

        #check picture is from camera or uplaod
        if image.shape == (450, 600, 3):
            # resize image to 192*256
            image = image[50:450, 150:450]
            image = cv2.resize(image, (192, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("image.jpg",image)
        else:
            cv2.imwrite("image.jpg",image)
    
        #run subprocess and load data to session
        run('python new_picture.py')
        try:    
            image = cv2.imread('image.jpg')
            parse = cv2.imread('parse.png', cv2.IMREAD_GRAYSCALE)
            with open("keypoint.json") as f:
                keypoint = json.load(f)
            session['keypoint'] = keypoint
            session['image'] = image
            session['parse'] = parse
            os.remove('image.jpg')
            os.remove('parse.png')
            os.remove('keypoint.json')
            return jsonify('OK')
        except:
            return jsonify('Error')

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
     
        # #get data from session
        image = session.get('image', 'not set')
        parse = session.get('parse', 'not set')
        keypoint = session.get('keypoint', 'not set')

        #call VITON
        arg = [cloth_path,image,parse,keypoint]
        with Pool(1) as p:
            result = p.map(Call_VITON,[arg])

        result = cv2.cvtColor(np.asarray(result[0]),cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (300, 400), interpolation=cv2.INTER_CUBIC)
        img_str = cv2.imencode('.jpg', result)[1].tostring()
        b64_code = str(base64.b64encode(img_str))
        b64_code = 'data:image/jpeg;base64,' + b64_code[2:-1]

    return jsonify({ 'image' : b64_code })

@app.route("/get")
def get():
    keypoint = session.get('keypoint', 'not set')
    image = session.get('image', 'not set')
    print(image, file=sys.stderr)
    # parse = session.get('parse', 'not set')
    return keypoint

@app.route("/check_session")
def check_session():
    keypoint = session.get('keypoint', 'not set')
    image = session.get('image', 'not set')
    parse = session.get('parse', 'not set')
    #print(type(keypoint), file=sys.stderr)
    if keypoint=='not set' or parse=='not set' or image=='not set':
        return '<script language=\'javascript\'>alert(\'Sorry system run error\'); window.location=\'demo\'</script>'
    else:
        return redirect('/tryon')

if __name__ == '__main__':
    app.debug = True
    app.run()
    # sess = Session()
    # sess.init_app(app)
    #print(, file=sys.stderr)