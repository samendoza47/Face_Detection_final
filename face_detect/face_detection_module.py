import math
import cv2
import dlib
import os
import numpy as np
import face_recognition
import settings
from os import walk


# -----: Getting to know blink ratio
def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_blink_ratio(eye_points, facial_landmarks):
    # loading all the required points
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    # calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


def retrive_all():
    cwd = os.getcwd()
    filenames = next(walk(os.path.join(cwd, "source", )), (None, None, []))[2]

    def function(k):
        return '.jpeg' in k or '.jpg' in k

    return list(filter(function, filenames))


def encode_all_faces():
    source_list = retrive_all()
    known_face_encodings.clear()
    known_face_names.clear()
    for source in source_list:
        source_name = source.split('.')[0]
        cwd = os.getcwd()
        completeName = os.path.join(cwd, "source", source)
        loaded_image = face_recognition.load_image_file(completeName)
        face_encoding = face_recognition.face_encodings(loaded_image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(source_name)
    print("loaded faces of ", known_face_names)


def release_camera():
    # Release handle to the webcam
    camera.release()
    cv2.destroyAllWindows()


camera = cv2.VideoCapture(0)
BLINK_RATIO_THRESHOLD = 5.7
known_face_encodings = []
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_names = []


def gen_frames():
    camera = cv2.VideoCapture(0)
    # Define how many faces we want to detect
    detector = dlib.get_frontal_face_detector()

    # -----Step 4: Detecting Eyes using landmarks in dlib-----
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # these landmarks are based on the image above
    left_eye_landmarks = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]

    # variable for blink detection and face detection
    blink_counter = False
    face_match = False

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break

        # detecting faces in the frame
        faces, _, _ = detector.run(image=frame, upsample_num_times=0, adjust_threshold=0.0)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # region Blink Detection Code
        # ----- Detecting Eyes using landmarks in dlib-----
        for face in faces:

            landmarks = predictor(frame, face)
            # -----: Calculating blink ratio for one eye-----
            left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
            right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
            blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blink_ratio > BLINK_RATIO_THRESHOLD:
                # Blink detected! Do Something!
                # Validate if the person has blinked
                blink_counter = True
                cv2.putText(frame, "BLINKING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                print(f'Blink action: {blink_counter}')
                settings.blick_detect_on_camera = True
                print(settings.blick_detect_on_camera)

        # end region

        # region
        # Only process every other frame of video to save time
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Not Authorized"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_match = True

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            settings.face_detect_on_camera = True
            print(settings.face_detect_on_camera)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
