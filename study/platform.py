import multiprocessing as mp
import cv2
import time
import msvcrt

def handleRtspVideo(id, q):
    #cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    cap = cv2.VideoCapture(id)
    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 1 else None

def handlDisplayVideo(q, p):
    #cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow('Test', frame)
        p.put(frame)
        p.get() if p.qsize() > 1 else None
        cv2.waitKey(1)

def handlFaceRecognition(p):
    while True:
        frame = p.get()
        time.sleep(1)
        print("process one image")
    print("Done")


def runSingleCamera(idlist):  # single camera
    
    mp.set_start_method(method='spawn')  # init

    imgqueues = [mp.Queue(maxsize=2), mp.Queue(maxsize=2)]
    processes = [mp.Process(target=handleRtspVideo, args=(idlist[0], imgqueues[0],)),
                 mp.Process(target=handlDisplayVideo, args=(imgqueues[0], imgqueues[1])),
                 mp.Process(target=handlFaceRecognition, args=(imgqueues[1],))]

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]





# def runMultiCamera(idlist):
#     user_name, user_pwd = "admin", "password"
#     camera_ip_l = [
#         "192.168.1.169",
#         "192.168.1.170",
#     ]

#     mp.set_start_method(method='spawn')  # init

#     queues = [mp.Queue(maxsize=2) for _ in camera_ip_l]

#     processes = []
#     for queue, camera_ip in zip(queues, camera_ip_l):
#         processes.append(mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)))
#         processes.append(mp.Process(target=queue_img_get, args=(queue, camera_ip)))

#     [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
#     [process.start() for process in processes]
#     [process.join() for process in processes]


if __name__ == '__main__':
    rtsp_url = ['rtsp://Jerry:Alrac2018!@192.168.1.64:554/h265/ch1/main/av_stream']
    runSingleCamera(rtsp_url)
    # run_multi_camera(rtsp_url)
