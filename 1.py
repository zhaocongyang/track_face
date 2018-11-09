# 1，将bbox_last的定义改到循环的外围
# 2，第一次使用识别物体的那一帧图片进行识别，然后再读取下一帧
# 3，将没用的flag = 2 删除了
import cv2
import sys
import numpy as np
import torch
import numpy
# hello = tf.constant('Hello Tensorflow!')
# sess = tf.Session()
# print(sess.run(hello))

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
def detect():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('/Users/apple/Desktop/py_opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/apple/Desktop/py_opencv/haarcascade_eye.xml')
    while(True):
        ret, img = camera.read()

        #img = cv2.resize(img, (641, 445),interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 2)
            #roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(gray, 1.03, 5, 0, (50, 50))
            for (ex, ey, ew, eh) in eyes:
                img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0 ,0, 255),2)
        cv2.namedWindow('face')
        cv2.imshow("face", img)
        cv2.waitKey(60)


if __name__ == '__main__' :

    x = torch.rand(3,5)
    print(x)
    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[3]
    print (tracker_types[3])
    tracker = cv2.TrackerKCF_create()

    # if int(minor_ver) < 3:
    #     tracker = cv2.Tracker_create(tracker_type)
    # else:
    #     if tracker_type == 'BOOSTING':
    #         tracker = cv2.TrackerBoosting_create()
    #     if tracker_type == 'MIL':
    #         tracker = cv2.TrackerMIL_create()
    #     if tracker_type == 'KCF':
    #         tracker = cv2.TrackerKCF_create()
    #     if tracker_type == 'TLD':
    #         tracker = cv2.TrackerTLD_create()
    #     if tracker_type == 'MEDIANFLOW':
    #         tracker = cv2.TrackerMedianFlow_create()
    #     if tracker_type == 'GOTURN':
    #         tracker = cv2.TrackerGOTURN_create()

    # read video
    # video = cv2.VideoCapture("/Users/apple/Desktop/4.mp4")
    video = cv2.VideoCapture(0)
    # box inite
    bbox = (287, 23, 86, 320)
    bbox_last = (287, 23, 86, 320)

    face_cascade = cv2.CascadeClassifier('//Users/zhaocongyang/Downloads/ppp/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/zhaocongyang/Downloads/ppp/haarcascade_eye.xml')

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    # main while
    while True:

        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()

        # Define an initial bounding box
        # bbox = (287, 23, 86, 320)
        # bbox_last = (287, 23, 86, 320)

        # face_cascade = cv2.CascadeClassifier('/Users/apple/Desktop/py_opencv/haarcascade_frontalface_default.xml')
        # eye_cascade = cv2.CascadeClassifier('/Users/apple/Desktop/py_opencv/haarcascade_eye.xml')
        failure_detected_num = 0

        # success_detect = 1 faile detect ,and if success detect success_detect > 1 then tracting
        success_detect = 1
        while success_detect == 1:
            ok, frame = video.read()
            print ("find face")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # Uncomment the line below to select a different bounding box
            #bbox = cv2.selectROI(frame, False)
            for (x, y, w, h) in faces:
                bbox = (x, y, w, h)
                bbox_last = (x, y, w, h)
                print ("init bbox")
                print (bbox[0])
                print (bbox[1])
                print (bbox[2])
                print (bbox[3])
                success_detect = 2
            if success_detect > 1 :
                print ("successful see face")
                break
            else:
                cv2.waitKey(1)

        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)

        guiji_x = []
        guiji_y = []
        i = 0
        # flag = 1 is success tracking and if flag = 2 tracking thing disappear and we need detect again
        flag = 1
        while flag == 1:
            print ("tracking")

            # # Read a new frame
            # ok1, frame = video.read()
            # if not ok1:
            #     break

            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)# 先用识别目标的那一帧图像刷新一下bbox 然后在读取下一帧存在frame里，不做处理，下次循环再处理，

            # Read a new frame
            ok1, frame = video.read()
            if not ok1:
                break

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # detect eyes and draw
            # eyes = eye_cascade.detectMultiScale(frame, 1.03, 5, 0, (50, 50))
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0 ,0, 0),5)


            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                # save bbox
                bbox_last = bbox
                # cv2.rectangle(frame, (int(bbox[0])+30, int(bbox[1]) +100),  (int(bbox[0])+180, int(bbox[1]) +170), (0,0,0), 5, 1)
                # cv2.rectangle(frame, (int(bbox[0])+230, int(bbox[1]) +100),  (int(bbox[0])+380, int(bbox[1]) +170), (0,0,0), 5, 1)
                # display x,y
                cv2.putText(frame, "x =  " + str(int(bbox[0] + 0.5*bbox[2])), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(frame, "y =  " + str(int(bbox[1] + 0.5*bbox[3])), (100,130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                guiji_x.append(int(bbox[0] + 0.5*bbox[2]))
                guiji_y.append(int(bbox[1] + 0.5*bbox[3]))
                i = i + 1
                if i < 100 :
                    for m in range(i):
                        cv2.circle(frame,(int(guiji_x[m]),int(guiji_y[m])),3,(55,255,155),3) # draw circle
                else:
                    i = 0
                    guiji_y = []
                    guiji_x = []

            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                failure_detected_num  = failure_detected_num + 1
                if failure_detected_num < 50:
                    p1 = (int(bbox_last[0]), int(bbox_last[1]))
                    p2 = (int(bbox_last[0] + bbox_last[2]), int(bbox_last[1] + bbox_last[3]))
                    cv2.rectangle(frame, p1, p2, (0,0,0), 2, 1)
                    cv2.circle(frame,(int(bbox_last[0] + 0.5*bbox_last[2]),int(bbox_last[1] + 0.5*bbox_last[3])),int(0.5*bbox_last[2]),(55,255,155),3)
                else:
                    flag = 2
            # flag = 2 break this while
            # if flag == 2:
            #     break

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

    cv2.waitKey(1)
