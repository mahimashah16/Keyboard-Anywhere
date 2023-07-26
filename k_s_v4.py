#import libraries
import cv2
import numpy as np




final=np.load('outfile.npy')
#print(final)

ans=["1","4","7","0","2","5","8","BAC","3","6","9","REF"]

#read image and convert it to grascale
image = cv2.imread("./dataset/B_9.jpg")
image = cv2.resize(image,(640,480))
cv2.imshow("original",image)

lower=np.array([0,58,30],dtype="uint8")
upper=np.array([33,255,255],dtype="uint8")


converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

	# show the skin in the image along with the mask
cv2.imshow("images", skinMask)

contours, hier = cv2.findContours(skinMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# find contours from mask
cv2.drawContours(image,contours,-1,(0,0,255),2)
#contours = contours[0] if len(contours) == 2 else contours[1]
c=max(contours,key=cv2.contourArea)
top = tuple(c[c[:,:,1].argmin()][0])
cv2.circle(image,top,8,(255,0,0),-1)

for i in range(0,13):
	if top[0] in range(final[0,i],final[0,i]+final[2,i]) and top[1] in range(final[1,i],final[1,i]+final[3,i]):
		print("Number you entered: {}".format(ans[i]))
		break

cv2.imshow("blob",image)
cv2.waitKey(0)