#import libraries
import cv2
import numpy as np
import pytesseract
from tempfile import TemporaryFile


def sort_cntr(x,y,w,h,cx,cy):
	x1=x
	y1=y
	w1=w
	h1=h
	cx1=cx
	cy1=cy
	S=np.asarray(sorted(range(len(x1)),key=lambda k:x1[k]))
	#print(x)
	S1=S[0:4]
	x31=x[np.array(S1)]
	y31=y[np.array(S1)]
	s31=np.asarray(sorted(range(len(y31)),key=lambda k:y31[k]))
	fs1=S1[np.array(s31)]
	#print(fs1)
	S2=S[4:8]
	x41=x[np.array(S2)]
	y41=y[np.array(S2)]
	s41=np.asarray(sorted(range(len(y41)),key=lambda k:y41[k]))
	fs2=S2[np.array(s41)]
	#print(fs2)
	S3=S[8:12]
	x32=x[np.array(S3)]
	y32=y[np.array(S3)]
	s32=np.asarray(sorted(range(len(y32)),key=lambda k:y32[k]))
	fs3=S3[np.array(s32)]
	#print(fs3)
	fS=np.concatenate([fs1,fs2,fs3],axis=0)
	a=x[np.array(fS)]
	b=y[np.array(fS)]
	c=w[np.array(fS)]
	d=h[np.array(fS)]
	e=cx[np.array(fS)]
	f=cy[np.array(fS)]
	return a,b,c,d,e,f

outfile = TemporaryFile()


#read image and convert it to grascale
image = cv2.imread("D:\keyboard\dataset\B.jpg")
image = cv2.resize(image,(640,480))
cv2.imshow("original",image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Thresholding
th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
th = cv2.GaussianBlur(th,(5,5),0)

cv2.imshow("thresholding",th)
contours, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# find contours from mask
i=0;

x=np.zeros((1,12),dtype=int)
y=np.zeros((1,12),dtype=int)
w=np.zeros((1,12),dtype=int)
h=np.zeros((1,12),dtype=int)
cX=np.zeros((1,12),dtype=int)
cY=np.zeros((1,12),dtype=int)
#cv2.drawContours(image,contours,-1,(0,0,255),2)
for cnt in contours:
	cnt_len= cv2.arcLength(cnt,True)
	cnt=cv2.approxPolyDP(cnt, 0.04 * cnt_len, True)
	if cv2.contourArea(cnt) > 9000  and cv2.contourArea(cnt)<100000 and len(cnt)==4 :#and cv2.contourArea(cnt) > 100 and cv2.isContourConvex(cnt):
		(x1, y1, w1, h1) = cv2.boundingRect(cnt)
		#print(x1,y1,w1,h1)
		#cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
		#cv2.putText(image,"#{}".format(i),(x1+50,y1+50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)
		x[0,i]=x1
		y[0,i]=y1
		w[0,i]=w1
		h[0,i]=h1
		M=cv2.moments(cnt)
		cX[0,i]=int(M["m10"]/M["m00"])
		cY[0,i]=int(M["m01"]/M["m00"])
		i=i+1
(fx,fy,fw,fh,fcx,fcy)=sort_cntr(x[0],y[0],w[0],h[0],cX[0],cY[0])
final=np.zeros((4,12),dtype=int)
final[0,:]=fx
final[1,:]=fy
final[2,:]=fw
final[3,:]=fh
print(final)
np.save('outfile',final)
for i in range(len(fx)):		
		cv2.putText(image,"#{}".format(i+1),(fcx[i]-20,fcy[i]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,225,0),2)
		cv2.rectangle(image, (fx[i], fy[i]), (fx[i] + fw[i], fy[i] + fh[i]), (225, 255, 0), 1)
		#cv2.putText(img,"#{}".format(i+1),(fcx[0,i]-20,fcy[0,i]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),2)
		#cv2.rectangle(img, (fx[0,i], fy[0,i]), (fx[0,i] + fw[0,i], fy[0,i] + fh[0,i]), (0, 255, 0), 1)
		cv2.imshow("squares",image)
		cv2.waitKey(0)