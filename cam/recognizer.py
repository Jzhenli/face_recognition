from multiprocessing import Process
from cam.models import dlib_model, classifier_model
import pickle
import dlib
import cv2
import numpy as np
import time

def get_time():
    localtime = time.localtime()
    capturetime = time.strftime("%Y-%m-%d %H:%M:%S", localtime)
    return capturetime


class FaceRecognition(Process):
    def __init__(self, img_queue, rst_queue):
        super().__init__()
        self.img_queue = img_queue
        self.rst_queue = rst_queue
        
    def init(self):
        # face detector
        # face_detector = dlib.cnn_face_detection_model_v1(dlib_model.cnn_face_detector_model_location())
        self.face_detector = dlib.get_frontal_face_detector()
        # face recognition
        self.sp = dlib.shape_predictor(dlib_model.pose_predictor_model_location())
        self.face_rec = dlib.face_recognition_model_v1(dlib_model.face_recognition_model_location())

    def run(self):
        self.init()

        with open(classifier_model.classifier_location(), 'rb') as classifier:
            (model, name_lib, feature_lib) = pickle.load(classifier)

        while True:
            try:
                frame = self.img_queue.get()
            except:
                pass
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame_dets =  self.face_detector(frame_rgb, 0)
                
                feature_lists=[]
                if len(frame_dets) :
                    for i, det in enumerate(frame_dets):
                        if type(det) == dlib.rectangle:
                            shape = self.sp(frame_rgb, det)
                        else:
                            shape = self.sp(frame_rgb, det.rect)
                        face_descriptor = self.face_rec.compute_face_descriptor(frame_rgb, shape, 1)
                        feature_lists.append(face_descriptor)
                if len(feature_lists):
                    predictions = model.predict(feature_lists)
                    namestr = ''
                    for idx, targ in enumerate(predictions):
                        distance = np.linalg.norm(np.array(feature_lists[idx]) - np.array(feature_lib[targ]))
                        namestr += (name_lib[targ]+' ' if distance < 0.6 else 'unknown ' )
                    rst = get_time() + ":" + namestr
                    print(rst)
                    try:
                        self.rst_queue.put_nowait(namestr)
                    except:
                        self.rst_queue.get_nowait()
                        self.rst_queue.put_nowait(namestr)
