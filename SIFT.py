import numpy as np
import cv2 as cv
import glob, os

#Scale-Invariant Feature Transform(SIFT)
input_img = cv.imread('BinaryFingerprint.jpg')#Loading image
input_img=input_img.astype('uint8')
gray= cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)#Transform into gray

sift = cv.xfeatures2d.SIFT_create()#create sift object
kp = sift.detect(input_img,None)#finding keypoints
img1=cv.drawKeypoints(input_img,kp,input_img)#drawing the keypoints
flag=0

os.chdir("./database")#folder with the fingerprints

for file in glob.glob("*.jpg"):
    
    frame=cv.imread(file)#loading image from the folder
    frame=frame.astype('uint8')
    gray1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()#create sift object for the image from the folder
    kp = sift.detect(frame,None)#finding keypoints for it
    img2=cv.drawKeypoints(frame,kp,frame)
    kp1, des1 = sift.detectAndCompute(img1,None)#list of the keypoints,(Number of Keypoints)×128
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1#using flann for finding hits
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 20)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    good = []
    for m,n in matches:#finding hits
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>=15:#more than 15 hits means its the fingerprint of the same person
        print("Matched "+str(file))
        print(len(good))
        flag=1
    else:
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0),matchesMask = matchesMask,flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv.imshow("Match",img3)

    cv.waitKey(0)
    cv.destroyAllWindows()

if flag==0:
    print("No Matches among the given set!!")
