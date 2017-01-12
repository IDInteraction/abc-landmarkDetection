#Import required modules
import cv2
import dlib
import sys

#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
video_capture = cv2.VideoCapture(sys.argv[1])
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("/usr/share/dlib/shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

while True:
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)

	detections, scores, idx =  detector.run(clahe_image, 1, 0) #Detect the faces in the image
        print scores, idx
	for k,d in enumerate(detections): #For each detected face
                cv2.rectangle(frame, (d.left(), d.bottom()), (d.right(), d.top()),(255,0,0), thickness=1)
		shape = predictor(clahe_image, d) #Get coordinates

		for i in range(1,68): #There are 68 landmark points on each face
			cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1) 

	cv2.imshow("image", frame) #Display the frame

	if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
		break


