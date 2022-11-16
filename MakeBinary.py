import cv2
import numpy as np


def main():
	img = cv2.imread("fingerprint.jpg",1)
        
	ret,thresh = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
	cv2.imshow("threshold",thresh)
	cv2.imwrite("./BinaryFingerprint.jpg",thresh)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__=='__main__':
	main()
