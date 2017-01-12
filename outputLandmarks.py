# Output landmark features for the bounding boxes given for each frame
# in a video file

#Import required modules
import cv2
import dlib
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Use dlib's landmark detector to calculate facial features and output their positions")
parser.add_argument("--infile", dest="infile", type=str, required=True)
parser.add_argument("--outfile", dest = "outfile", type = str, required = True)
parser.add_argument("--bboxes", dest = "bboxes", type = str, required = False)
parser.add_argument("--showvideo", dest = "showvideo", action = "store_true")
parser.set_defaults(showvideo = False)
parser.add_argument("--skipframe", dest = "skipframe", type=int, action="store")
parser.add_argument("--skipms", dest = "skipms", type = int, action = "store")

args = parser.parse_args()

if args.showvideo == True:
    cv2.namedWindow("landmark")

video_capture = cv2.VideoCapture(args.infile)

if(args.skipframe and args.skipms):
	print("Cannot skip frames and time")
	quit()


#rect = dlib.rectangle( left = 290, right = 333, top = 127, bottom = 84)



if(args.skipframe):
	print("Skipping " + str(args.skipframe) + " frames")
	video_capture.set(cv2.CAP_PROP_POS_FRAMES, args.skipframe)
if(args.skipms):
	print("Skipping " + str(args.skipms) + " ms")
	video_capture.set(cv2.CAP_PROP_POS_MSEC, args.skipms)

if  not args.bboxes:
	print("Haven't yet implemented face detection")
	quit()

bboxes = pd.read_csv(args.bboxes, skiprows = 1, names=["frame",
                                            "time",
                                            "actpt",
                                            "bbcx",
                                            "bbcy",
                                            "bbw",
                                            "bbh",
                                            "bbrot",
                                            "bbv1x",
                                            "bbv1y",
                                            "bbv2x",
                                            "bbv2y",
                                            "bbv3x",
                                            "bbv3y",
                                            "bbv4x",
                                            "bbv4y"])


detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("/usr/share/dlib/shape_predictor_68_face_landmarks.dat")

while True:
	ret, img = video_capture.read()
	frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# This improves contrast, but seems to amplify compression noise
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	#clahe_image = clahe.apply(gray)

	#detections = detector(clahe_image, 1) #Detect the faces in the image

	# Create bounding box rectangle from loaded file

	thisframe = bboxes[bboxes['frame'] == frame].iloc[0]
	if(thisframe['bbrot'] != 0):
		print("Bounding box rotation must be 0")
		quit()

	bbox = dlib.rectangle( left = long(thisframe.bbv1x), \
	right = long(thisframe.bbv3x), top = long(thisframe.bbv3y), \
	bottom = long(thisframe.bbv1y))

	print frame

	shape = predictor(gray, bbox) #Get coordinates
	for i in range(1,68): #There are 68 landmark points on each face
		cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1)
	cv2.rectangle(img, (int(bbox.left()), int(bbox.top())), (int(bbox.right()), int(bbox.bottom())), color = (255,255,255), thickness=1)
	cv2.imshow("image", img) #Display the frame

	if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
		break
