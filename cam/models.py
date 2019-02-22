# -*- coding: utf-8 -*-
#from pkg_resources import resource_filename

class dlib_model:
	def pose_predictor_model_location():
	    return "./models/dlib/shape_predictor_68_face_landmarks.dat"


	def pose_predictor_five_point_model_location():
	    return "./models/dlib/shape_predictor_5_face_landmarks.dat"


	def face_recognition_model_location():
	    return "./models/dlib/dlib_face_recognition_resnet_model_v1_for_asian.dat"


	def cnn_face_detector_model_location():
	    return "./models/dlib/mmod_human_face_detector.dat"

class opencv_model:
	def caff_model_location():
	    return "./models/opencv/res10_300x300_ssd_iter_140000_fp16.caffemodel"

	def caff_cfgfile_location():
	    return "./models/opencv/deploy.prototxt"

	def tensorflow_model_location():
		return "./models/opencv/opencv_face_detector_uint8.pb"

	def tensorflow_cfgfile_location():
		return "./models/opencv/opencv_face_detector.pbtxt"

class classifier_model:
	def classifier_location():
		return "./models/classifier/face_classifier.pkl"
