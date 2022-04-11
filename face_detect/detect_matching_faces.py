import os

import face_recognition
from os import walk



def compare_images(source, target):

    known_image = face_recognition.load_image_file(source)
    known_encoding = face_recognition.face_encodings(known_image ,known_face_locations=None, num_jitters=1, model='small')[0]
    unknown_image = face_recognition.load_image_file(target)
    unknown_encoding = face_recognition.face_encodings(unknown_image,known_face_locations=None, num_jitters=1, model='small')[0]
    print("Comapring",source,target )
    results = face_recognition.compare_faces([known_encoding], unknown_encoding,tolerance=0.6)
    print("result is", results)
    return results[0]

def retrive_all():
    cwd = os.getcwd()
    filenames = next(walk(os.path.join(cwd, "source",)), (None, None, []))[2]

    def function(k):
        return '.jpeg' in k or '.jpg' in k

    return list(filter(function, filenames))

def is_matched(target):
    source_list=retrive_all()
    output="NOT FOUND"
    for source in source_list:
        source_name=source.split('.')[0]
        cwd = os.getcwd()
        completeName = os.path.join(cwd, "source", source)
        result=compare_images(completeName, target)
        if result :
            output = "MATCH FOUND FOR " + source_name
            break

    print(output)
    return output