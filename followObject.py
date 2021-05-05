import inspect

import cv2 as cv
import numpy as np
import math

points = []
keypoints = None
good_matches = None
medx = []
medy = []
medw = []
medh = []
cnts = []

'''
This is a simple method to loop into interesting object. Its used to find values od train indekses-> 
is appending the points list with values from img2(video frames) coresponding to given image img1. 
'''


def print_members(obj):
    global points
    for name, value in inspect.getmembers(obj):
        try:
            if name == "trainIdx":
                points.append(value)
            float(value)
            print(f'{name} -> {value}')
        except Exception:
            pass


'''
This is a method which as arguments takes descriptor of base picture and descriptor of each frame and by BFMatcher
matches the keypoints as the result we get the good_matches array with 10 best matches.
'''


def match_frame(des1, des2):
    global good_matches

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # im mniejszy dystans tym lepszy
    good_matches = matches[:10]

    for i in good_matches:
        print_members(i)


'''
This is a method where as arguments is given image where we draw keypoints and main frame from video.
In the method drawRec we add mask and we draw rectangle around average from x,y points taken based
on color of keypoints. There is also a condition: if there is less than 10 drawn keypoints we don't draw rec. 

'''


def drawRec(keypoints, frame):
    global medy, medx, medw, medh, cnts
    mask = cv.inRange(keypoints, (0, 0, 230), (0, 0, 255))
    res = cv.bitwise_and(keypoints, keypoints, mask=mask)
    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    ret, tframe = cv.threshold(res, 0, 255, cv.THRESH_BINARY)

    (cnts, _) = cv.findContours(tframe.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        print(cnt)
        xi, yi, w, he = cv.boundingRect(cnt)
        medx.append(xi)
        medy.append(yi)
        medw.append(w)
        medh.append(he)
    loop = False
    if len(medx) > 10:
        loop = True
        avx = int(np.average(medx))
        avy = int(np.average(medy))
        avw = int(np.average(medw))
        avh = int(np.average(medh))
    lengX = len(medx)
    lengY = len(medy)
    if loop:
        cv.rectangle(frame, (avx - avw * lengX * 15, avy - avh * lengY * 15),
                     (avx + avw * lengX * 15, avy + avh * lengY * 15), (0, 255, 0), 4)


'''
This is a min method where  is all main code as:
base image read, video read, orb detection from image1 and frames from video, usage of match_frame method,
displaying frame with drawn rectangle, 
'''


def main():
    global keypoints, good_matches, points, medh, medw, medx, medy, cnts

    img1 = cv.imread('pila1.jpg')
    orb = cv.ORB_create()
    img1gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(img1gray, None)

    video = cv.VideoCapture('pila_film.mp4')
    while video.isOpened():
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv.waitKey(-1)

        _, frame = video.read()

        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        kpFrame, desFrame = orb.detectAndCompute(frameGray, None)
        match_frame(des1, desFrame)

        p = []
        for i in points:
            p.append(kpFrame[i])

        keypoints = cv.drawKeypoints(frame, p, None, color=(0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        drawRec(keypoints, frame)
        cv.namedWindow("detectedImage", cv.WINDOW_GUI_NORMAL)
        cv.imshow("detectedImage", frame)

        medy.clear()
        medx.clear()
        p.clear()
        points.clear()
        cnts.clear()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
