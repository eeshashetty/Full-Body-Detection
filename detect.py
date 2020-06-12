import cv2
import argparse
import imutils
import concurrent.futures

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

#parser = argparse.ArgumentParser()
#parser.add_argument('--path', help='0 for livestream, provide path otherwise - default is 0', default=0)
#args = parser.parse_args()

#path = 0 if args.path=='0' else args.path
path = 'rtsp://admin:admin@192.168.0.100/1'
cap = cv2.VideoCapture(path)

cascade_full = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cascade_upper = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cascade_side = cv2.CascadeClassifier('haarcascade_profileface.xml')

with concurrent.futures.ThreadPoolExecutor() as executor:
	while True:
		future = executor.submit(cap.read)

		ret, frame = future.result()
		if ret:
			#det_frame = detect_person(frame)
			future2 = executor.submit(detect_person, frame)
			cv2.imshow('Footage', future2.result())
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		else:
			break

cap.release()
cv2.destroyAllWindows()
