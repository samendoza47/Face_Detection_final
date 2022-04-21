import base64

from flask import Flask, render_template, Response, request, url_for, jsonify, send_file
import os
import face_detection_module as fbt
import settings
from werkzeug.utils import secure_filename, redirect, send_from_directory


app = Flask(__name__)
settings.init()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_names = []


@app.route('/index')
@app.route('/')
def index():
    fbt.encode_all_faces()
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(fbt.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image-upload-view')
def image_upload_view():
    fbt.release_camera()
    return render_template('imageupload.html')


@app.route('/image-upload', methods=['POST'])
def image_upload():
    i = request.files['image']  # get the image
    fileName = request.form["personName"]
    f = ('%s.jpeg' % fileName)
    # main folder
    cwd = os.getcwd()
    completeName = os.path.join(cwd, "source")
    i.save('%s/%s' % (completeName, f))

    return Response("%s saved" % f)


@app.route('/display/<filename>')
def display_image(filename):
    cwd = os.getcwd()
    completeName = os.path.join(cwd, "source",filename)
    return send_file(completeName, mimetype='image/jpg')

@app.route('/view-all-users')
def retrive_all_users():
    values= fbt.retrive_all()
    return render_template('viewAllImages.html', values=values)



@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
