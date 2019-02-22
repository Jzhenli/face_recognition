#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import dlib
import cv2
import pickle
from sklearn.svm import SVC
from cam.models import dlib_model, classifier_model


FaceDetection = dlib.get_frontal_face_detector()
# face recognition
sp = dlib.shape_predictor(dlib_model.pose_predictor_model_location())
Description = dlib.face_recognition_model_v1(dlib_model.face_recognition_model_location())

def check_parameter(param, param_type, create_new_if_missing=False):
    assert param_type == 'file' or param_type == 'directory'
    if param_type == 'file':
        assert os.path.exists(param)
        assert os.path.isfile(param)
    else:
        if create_new_if_missing is True:
            if not os.path.exists(param):
                os.makedirs(param)
            else:
                assert os.path.isdir(param)
        else:
           assert os.path.exists(param)
           assert os.path.isdir(param)

def listdir(top_dir, type='image'):
    tmp_file_lists = os.listdir(top_dir)
    file_lists = []
    if type == 'image':
        for e in tmp_file_lists:
           if e.endswith('.jpg') or e.endswith('.png') or e.endswith('.bmp') or e.endswith('.JPG'):
               file_lists.append(e)
    elif type == 'dir':
        for e in tmp_file_lists:
           if os.path.isdir(top_dir + e):
              file_lists.append(e)
    else:
        raise Exception('Unknown type in listdir')
    return file_lists

def imread(f):
    return cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)

def extract_feature(src_dir, type_name):
    check_parameter(src_dir, 'directory')
    if src_dir[-1] != '/':
        src_dir += '/'
    image_lists = listdir(src_dir, type_name)
    names_lists= []
    feature_lists=[]
    labels =[]
    for n,e in enumerate(image_lists):
        img=imread(''.join([src_dir,e]))
        dets=FaceDetection(img,1) #upsampling
        if len(dets)==0:
            print("The faces of {}  >>>>>>>>>>  detecting defeat".format(e))
        else:
            params=e.split(".")
            name_id=params[0]
            print("ID : {} , Name : {}  ".format(n,name_id)) 
            Best_face=max(dets, key=lambda rect: rect.width() * rect.height())
            shape = sp(img, Best_face)
            face_descriptor=Description.compute_face_descriptor(img, shape, 10)
            feature_lists.append(face_descriptor)
            names_lists.append(name_id)
            labels.append(n)

    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(feature_lists, labels)

    # Saving classifier model
    with open(classifier_model.classifier_location(), 'wb') as outfile:
        pickle.dump((model, names_lists, feature_lists), outfile)
    print('Saved classifier model to ', classifier_model.classifier_location())

def run():
    print("wait for system init...")
    extract_feature('./images','image')
    print("training finished.")

if __name__ == '__main__':
    run()












