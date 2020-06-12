# USAGE
# python read_frames_fast.py --video videos/jurassic_park_intro.mp4

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())

cascade_full = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cascade_upper = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cascade_side = cv2.CascadeClassifier('haarcascade_profileface.xml')


def detect_person(frame):
	im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	im = cv2.equalizeHist(im)
	
	# To Detect Side Faces
	faces = cascade_side.detectMultiScale(im, 1.3)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)
	
	# Full Body Detection
	full_people = cascade_full.detectMultiScale(im, 1.3)
	
	for (x,y,w,h) in full_people:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
	
	# Upper Body Detection
	half_people = cascade_upper.detectMultiScale(im, 1.3)
	for (x,y,w,h) in half_people:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)


	return frame

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

# loop over frames from the video file stream
while fvs.more():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
	frame = fvs.read()
    
    frame = detect_person(frame) 

	#frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])

	# display the size of the queue on the frame
	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

	# show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()