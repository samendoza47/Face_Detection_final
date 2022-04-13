from flask import Flask, render_template, Response, request
import os
import face_detection_module as fbt
import settings


app = Flask(__name__)
settings.init()


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_names = []


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


@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
