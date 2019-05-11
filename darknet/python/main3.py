import numpy as np
from matplotlib import pyplot as plt
import cv2
from darknet import PlayerDetector
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
from multiprocessing import Process, Pipe, Pool
from functools import partial
from threading import Thread
from concurrent.futures.thread import ThreadPoolExecutor
import datetime
from numpy.linalg import norm

def get_full_line(p1, p2):
	slope = (p2[1]-p1[1])/float(p2[0]-p1[0])
	init_img = [10000,10000]
	end_img = [-10000,-10000]

	init_img[1] = int(-(p1[0]-init_img[0]) * slope + p1[1])
	end_img[1] = int(-(p2[0]-end_img[0]) * slope + p2[1])
	return tuple(init_img), tuple(end_img)

def intersection(line1, line2):
	p1, p2, p3, p4 = line1['p1'], line1['p2'], line2['p1'], line2['p2']
	x = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[0]-p4[0]) - (p1[0]-p2[0]) * (p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0]-p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]-p4[0]))
	y = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0]-p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]-p4[0]))
	return (x,y)

def update_bounding_box(frame, tracker, send_end):
	(img_h, img_w) = frame.shape[:2]
	send_end.send(tracker.update(frame))

def same_lines(line1, line2):
	rho1,theta1 = line1
	rho2,theta2 = line2
	return abs(rho1-rho2) < 20 and abs(theta1-theta2) < 0.1
	

BOUNDING_BOX_END_LIMIT = 100
CSRT_PARAMS = cv2.FileStorage("CSRT_params.json", cv2.FileStorage_READ)
KCF_PARAMS = cv2.FileStorage("KCF_params.json", cv2.FileStorage_READ)
RANDOM_FIELD_POINT = (280,501)

cap = cv2.VideoCapture('../video.mp4')

out = cv2.VideoWriter('../output.mp4', 0x7634706d, 30.0, (1280,720))

if (not cap.isOpened()):
	raise Exception('Video cound not be opened')

#player_detector = PlayerDetector()

start_frame = 0
end_frame = 300
for frame_index in range(0, end_frame):
	print(frame_index)
	if frame_index<start_frame:
		continue

	ret, frame = cap.read()
	heigth, width = frame.shape[:2]
	if ret==False:
		break
	
	frame_copy = frame.copy()
	

	if frame_index==start_frame:
		
		
		### Deteccion linea offside
		'''
		plt.imshow(frame)
		points = plt.ginput(100, show_clicks=True)
		assert len(points)%2==0, 'Points count should be pair'
		assert len(points)>=4, 'Should be at least two lines'

		print(points)
		lines = []
		for i in range(0, len(points), 2):
			line = {'p1': points[i], 'p2': points[i+1]}
			lines.append(line)

		print(lines)

		lines_intersection = []
		for i in range(len(lines)-1):
			intersec = intersection(lines[i], lines[i+1])
			lines_intersection.append(intersec)

		x_intersection = list(map(lambda x: x[0], lines_intersection))
		y_intersection = list(map(lambda x: x[1], lines_intersection))
		vp = (np.median(x_intersection), np.median(y_intersection))
		xs = [283, vp[0]]
		ys = [501, vp[1]]
		plt.plot(xs,ys,linewidth=2)
		raw_input("Press Enter to continue")
		'''
		
		### Deteccion jugadores
		

		#yolo_img = player_detector.open_img(frame)
		#res = player_detector.detect_players(yolo_img)
		res = [('person', 0.9896788001060486, (367.18341064453125, 437.80145263671875, 38.514793395996094, 69.6651611328125)), ('person', 0.9813730120658875, (291.6535949707031, 585.266357421875, 47.23373031616211, 89.79061889648438)), ('person', 0.9451802372932434, (1184.445068359375, 438.14361572265625, 27.63399887084961, 70.66059875488281)), ('person', 0.9427978992462158, (613.549072265625, 310.98052978515625, 28.87258529663086, 47.110721588134766)), ('person', 0.9319689273834229, (1070.2806396484375, 212.15728759765625, 16.6505184173584, 33.18511962890625)), ('person', 0.9279605746269226, (298.9461669921875, 471.6380310058594, 60.612213134765625, 70.96109008789062)), ('person', 0.8941689133644104, (917.6439208984375, 270.20245361328125, 17.11932373046875, 29.22058868408203)), ('person', 0.8926199078559875, (1047.6141357421875, 487.6801452636719, 26.599294662475586, 70.4723892211914)), ('person', 0.8923733830451965, (688.3287353515625, 300.20904541015625, 17.36338233947754, 54.863426208496094)), ('person', 0.8835442066192627, (1109.42041015625, 202.57579040527344, 17.15543556213379, 30.07115936279297)), ('person', 0.8802978992462158, (828.4103393554688, 457.4922180175781, 54.69183349609375, 60.64540100097656)), ('person', 0.8713310360908508, (856.3143920898438, 386.87445068359375, 37.859336853027344, 45.99755096435547)), ('person', 0.7614070773124695, (1156.033203125, 520.988525390625, 39.83382034301758, 65.85809326171875)), ('person', 0.7554710507392883, (530.2201538085938, 142.78485107421875, 15.13008975982666, 29.750947952270508))]
		
		## Saco las detecciones con poca probabilidad, o que no sean personas 
		res = [r for r in res if r[1] > 0.6 and r[0] == 'person'] 

		## Saco a los jugadores que estan muy cerca del borde
		res = [r for r in res if not (width - r[2][0] < BOUNDING_BOX_END_LIMIT or r[2][0] < BOUNDING_BOX_END_LIMIT)] 
		
		player_trackers = []

		
		for i in range(len(res)):
			bb = res[i]
			bounding_box = bb[2]
			x = int(bounding_box[0])
			y = int(bounding_box[1])
			w = int(bounding_box[2]/2)
			h = int(bounding_box[3]/2)
			
			#tracker = cv2.TrackerCSRT_create()
			#tracker.read(CSRT_PARAMS.getFirstTopLevelNode())
			tracker = cv2.TrackerKCF_create()
			tracker.read(KCF_PARAMS.getFirstTopLevelNode())
			tracker.init(frame, (x-w, y-h, w*2, h*2))
			player_trackers.append(tracker)

		continue

	
	
	start_time = datetime.datetime.now()
	
	with ThreadPoolExecutor(max_workers=22) as executor:
		pipe_list = []
		#start_time = datetime.datetime.now()
		for tracker in player_trackers:
			
			recv_end, send_end = Pipe(False)
			executor.submit(update_bounding_box, frame, tracker, send_end)
			pipe_list.append(recv_end)
	

	
	updated_boxes = [x.recv() for x in pipe_list]
	
	
	player_track_to_remove = []
	bounding_boxes = []
	for i in range(len(updated_boxes)):
		(success, box) = updated_boxes[i]
		if success:
			(x, y, w, h) = [int(v) for v in box]
			## Saco a los jugadores que estan muy cerca del borde
			if width-x < BOUNDING_BOX_END_LIMIT or x < BOUNDING_BOX_END_LIMIT:
				player_track_to_remove.append(i)
			else:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				bounding_boxes.append((x, y, w, h))
	
	
	player_trackers = [tracker for i, tracker in enumerate(player_trackers) if i not in player_track_to_remove]

	end_time = datetime.datetime.now()	
	print('time: ', (end_time-start_time).total_seconds())

	


	gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray_frame,(5, 5),0)

	sobelx = cv2.Sobel(blur_gray,cv2.CV_8U,1,0,ksize=-1)
	#cv2.imshow('sobelx',sobelx)

	
	############ calculate vanishing point
	lines = cv2.HoughLines(sobelx,1,np.pi/180,200)
	
	parallel_lines = []
	i = 0
	j = 0
	
	for j in range(len(lines)):
		for i in range(len(lines[j])):
			if len(parallel_lines) == 2:
				break
			rho,theta = lines[j][i]
			if not (len(parallel_lines) == 1 and same_lines(lines[j][i], (first_rho,first_theta))):
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))	

				first_rho, first_theta = rho,theta
				parallel_lines.append({'p1': (x1,y1), 'p2': (x2,y2)})
				try:
					if len(parallel_lines) == 2 and intersection(parallel_lines[0], parallel_lines[1])[1] > 0:
						parallel_lines.pop()
				except:
					## lines are parallel
					parallel_lines.pop()

		
	x1,y1 = parallel_lines[0]['p1']
	x2,y2 = parallel_lines[0]['p2']
	cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

	x1,y1 = parallel_lines[1]['p1']
	x2,y2 = parallel_lines[1]['p2']
	cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

	vanishing_point = intersection(parallel_lines[0], parallel_lines[1])
	#cv2.line(frame,vanishing_point,RANDOM_FIELD_POINT,(255,0,0),2)

	############

	############ calculate leftmost player
	leftmost_player = None
	for i in range(len(bounding_boxes)):
		(x, y, w, h) = bounding_boxes[i]
		# asumo que atacan para la izquierda
		p = (x,y+h)
		if leftmost_player is None:
			leftmost_player = p
		direction = (p[0]-vanishing_point[0])*(leftmost_player[1]-vanishing_point[1])-(p[1]-vanishing_point[1])*(leftmost_player[0]-vanishing_point[0])
		if direction < 0:
			leftmost_player = p
	############

	############ draw offside_line
	offside_line = get_full_line(vanishing_point, leftmost_player)
	
	cv2.line(frame,vanishing_point,leftmost_player,(255,255,0),2)
	############

	out.write(frame)
	continue






cap.release()
out.release()

cv2.destroyAllWindows()