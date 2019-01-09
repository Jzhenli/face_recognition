import cv2
import time
from multiprocessing import Process, Queue
import face_recognition
import pickle

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

def get_time():
    localtime = time.localtime()
    capturetime = time.strftime("%Y-%m-%d %H:%M:%S", localtime)
    return capturetime

def predict(X_input_img, knn_clf=None, model_path=None, distance_threshold=0.6):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(X_input_img)
    
    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_input_img, known_face_locations=X_face_locations, num_jitters=2)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def clear_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except:
        pass
    queue.close()
    queue.join_thread()


    
def FaceDetect(frame_queue, result_queue):
    while True:
        try:
            inputframe = frame_queue.get_nowait()
            if inputframe is None:
                break
            predictions = predict(inputframe, model_path="trained_knn_model.clf")
            name_list = [name for name, (top, right, bottom, left) in predictions]
            if len(name_list):
                print(get_time(), ":", ','.join(name_list))
                while not result_queue.empty():
                    result_queue.get_nowait()
                result_queue.put(name_list)
        except:
            pass

    print("Detection is done.")

def main():
    frame_queue = Queue(4)
    result_queue = Queue(1)

    face_proc = Process(target=FaceDetect, args=(frame_queue, result_queue))
    face_proc.start()



    scale = 4
    # Get a reference to webcam #0 (the default one)
    cap = cv2.VideoCapture(0)
        # id = 'rtsp://Jerry:Alrac2018!@192.168.1.64:554/h265/ch1/sub/av_stream'
    # cap = cv2.VideoCapture(id)
    
    cap.set(3, 640) #set width
    cap.set(4, 480) #set height
    count = 0
    while cap.isOpened():
        # Grab a single frame of video
        ret, frame = cap.read()
        # Resize frame of video to 1/scale size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1.0/scale, fy=1.0/scale)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # Display the results
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/scale size
            # Draw a box around the face
            cv2.rectangle(frame, (left*scale, top*scale), (right*scale, bottom*scale), (0, 0, 255), 2)
        
        # bordered_image = cv2.copyMakeBorder(frame, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # try:
            # name = result_queue.get_nowait()
            # cv2.putText(bordered_image, ','.join(name), (20, 520), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 1)
        # except:
            # pass
        # Display the resulting image
        cv2.imshow('Video', frame)
        
        
        
        if count >= 4:
            count = 0
            try:
                frame_queue.put_nowait(small_frame)
            except:
                while not frame_queue.empty():
                    frame_queue.get()
        count += 1
        
        

        # 'ESC' for quit
        key = cv2.waitKey(1)
        if key == 27:
            while not frame_queue.empty():
                frame_queue.get_nowait()
            frame_queue.put(None)
            face_proc.join()
            clear_queue(frame_queue)
            clear_queue(result_queue)
            break
            
    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("OpenCV version: " + cv2.__version__)
    print("waiting for init... ")
    main()
    
