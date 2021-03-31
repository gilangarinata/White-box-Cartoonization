import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm
from PIL import Image


from rembg.bg import remove
import io

MODEL_PATH = './test_code/saved_models'
UPLOAD_FOLDER = './imageuploads'
SAVED_FOLDER = './images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'kuning77'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("./images", filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process/<isRemove>', methods=['POST','GET'])
def upload_file(isRemove):
    if request.method == 'POST':
        if 'file' not in request.files:
            return {
                "code" : 1210,
                "message" : "No file part" ,
                "path" : None
            }
        file = request.files['file']
        if file.filename == '':
            return {
                "code" : 1211,
                "message" : "no selected file",
                "path" : None 
            }
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = UPLOAD_FOLDER + '/' + filename
            model_path = MODEL_PATH
            save_folder = SAVED_FOLDER
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            cartoonize(img_path, save_folder, model_path,filename)
            path = "./images/" + filename
            f = np.fromfile(path)
            if isRemove == '0':
                respon = {
                    "code" : 2000,
                    "message" : "success",
                    "path" : request.base_url.replace("process","").replace("/0","").replace("/1","") + "uploads/" +filename
                }
                print(respon)
                return respon
            else:
                result = f
                img = Image.open(io.BytesIO(result)).convert("RGBA")
                img.save("./images/" + "final" +filename + ".png")
                respon = {
                    "code" : 2000,
                    "message" : "success",
                    "path" : request.base_url.replace("process","").replace("/0","").replace("/1","") + "uploads/" + "final" +filename + ".png"
                }
                print(respon)
                return respon
        return {
            "message" : "failed"
        }

def cartoonize(load_file, save_folder, model_path, name):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    try:
        save_path = os.path.join(save_folder, name)
        image = cv2.imread(load_file)
        image = resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, output)
    except IndexError as e:
        print('cartoonize {} failed'.format(load_file) + e)


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image

if __name__ == '__main__':
    app.run(host='127.0.0.1')