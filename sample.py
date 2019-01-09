import cv2
import os

def getImgFromCam(name):
    path_make_dir = './data/train/'+name
    if os.path.isdir(path_make_dir):
        pass
    else:
        os.mkdir(path_make_dir)
        
    print("Wait open cam, 's' save image, 'esc' exit.")
    cap = cv2.VideoCapture(0)
    windowName = "Face Sample"
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow(windowName, frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            cnt += 1
            image = path_make_dir + '/' + name + str(cnt) + ".jpg"
            cv2.imwrite(image, frame)
            print("saving: ", image)
        
        # 'ESC' for quit
        if key == 27:
            break
            
    cap.release()
    cv2.destroyWindow(windowName)


if __name__ == "__main__":
    print("OpenCV version: " + cv2.__version__)
    print("waiting for init... ")
    print("-------------------------------------")
    while True:
        name = input("what's your name: ")
        getImgFromCam(name)
        key = input("Continue to collect[y/n]: ")
        if key in ['n', 'N', 'no', 'No']:
            break
        print("-------------------------------------")
    print("sample finish.")