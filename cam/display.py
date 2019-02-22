import cv2
from cam.camera import WebcamVideoStream
from cam.recognizer import FaceRecognition
from cam.qarrays import ArrayQueue
from cam.models import opencv_model
from multiprocessing import Queue
import time

class DisplayVideo:
    def __init__(self, src=0, name="DisplayVideo"):
        self.name = name
        self.camid = src
        self.camera = WebcamVideoStream(self.camid).start()
        self.faceDetectorInit()
        
    def faceDetectorInit(self,DNN='TF'):
        if DNN == "CAFFE":
            modelFile = opencv_model.caff_model_location()
            configFile = opencv_model.caff_cfgfile_location()
            self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = opencv_model.tensorflow_model_location()
            configFile = opencv_model.tensorflow_cfgfile_location()
            self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    
    def detectFaceOpenCVDnn(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)
                y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)
                x2 = min(int(detections[0, 0, i, 5] * frameWidth), frameWidth)
                y2 = min(int(detections[0, 0, i, 6] * frameHeight), frameHeight)
                bboxes.append((x1, y1, x2, y2))
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes
    
    def display(self):
        windowName = 'Face Recognition V1.0'
        frame = self.camera.read()
        print(frame.shape)
        qimg = ArrayQueue(2*frame.size) #frame.dtype must be uint8
        qrst = Queue(maxsize=5)
        rec = FaceRecognition(qimg, qrst)
        rec.daemon = True
        rec.start()
        
        # loop over some frames...this time using the threaded stream
        cnt = 0
        rst = ''
        while True:
            frame = self.camera.read()
            outOpencvDnn, bboxes = self.detectFaceOpenCVDnn(self.net, frame, 0.7)
            overlay = outOpencvDnn.copy()
            cv2.rectangle(overlay, (frame.shape[1]-200, 0), (frame.shape[1], 80),(88, 88, 88), -1)
            try:
                rst = qrst.get_nowait()
            except:
                pass
            if len(bboxes):
                cv2.putText(overlay, rst, (frame.shape[1]-180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            dst = cv2.addWeighted(overlay, 0.5, outOpencvDnn, 0.5, 0)
            cv2.imshow(windowName, dst)
            try: 
                qimg.put(frame) if len(bboxes) and cnt%4==0 else None
            except:
                pass
            cnt += 1
            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty(windowName,1) == -1:
                qimg.clear()
                self.camera.stop()
                break

        # do a bit of cleanup
        cv2.destroyWindow(windowName)
        