import cv2, numpy
from numba import njit
from modules.prep_utils import IOU


@njit(parallel = False, fastmath = True)
def fast_select_detections(net_list, width, height, min_confidence, class_list):

		object_list = []
	
		for out_index in range(len(net_list)):
			for detection_index in range(len(net_list[out_index])):
	
				detection = net_list[out_index][detection_index]
	
				scores = detection[5:]
				raw_id = numpy.argmax(scores)
				confidence = scores[raw_id]
				class_id = 0

				if len(class_list) > raw_id: 
					class_id = class_list[raw_id]
				
				if confidence > min_confidence:
		
					center_x = detection[0]*width
					center_y = detection[1]*height
					box_w = detection[2]*width
					box_h = detection[3]*height
		
					x1 = center_x - (box_w/2)
					y1 = center_y - (box_h/2)
		
					x2 = center_x + (box_w/2)
					y2 = center_y + (box_h/2)
		
					object_list.append(((class_id, x1, y1, x2, y2), confidence, 0))
	
		return object_list



class Detector:
	def __init__(self, weight, config, min_confidence, nms_confidence, class_list, resolution):

		self.resolution = resolution

		self.min_confidence = min_confidence
		self.nms_confidence = nms_confidence

		self.class_list = class_list

		self.model = cv2.dnn.readNet(weight, config)
		self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

		self.layer_names = self.model.getLayerNames()
		self.output_layers = self.model.getUnconnectedOutLayersNames()

		test_image = cv2.imread("test.jpg")
		h, w, c = test_image.shape
		self.full_detection(test_image, w, h)


	def detect_objects(self, image):

		self.height, self.width, c = image.shape 

		blob = cv2.dnn.blobFromImage(image, 1/255, (self.resolution, self.resolution), (0, 0, 0), swapRB=True, crop=False)
		self.model.setInput(blob)
		net_list = self.model.forward(self.output_layers)

		return net_list

	def select_detections(self, net_list, width, height):
		return fast_select_detections(net_list, width, height, self.min_confidence, self.class_list)

	def nms(self, object_list):

		sorted_detections = list(reversed(sorted(object_list, key=lambda x:x[1])))
	
		overlap_objects = []
	
		for obj_n1 in range(len(sorted_detections)):
			for obj_n2 in range(len(sorted_detections)):
	
				if obj_n2 > obj_n1 and obj_n1 not in overlap_objects:
	
					box1 = sorted_detections[obj_n1][0]
					box2 = sorted_detections[obj_n2][0]
	
					if box1[0] == box2[0]:
	
						iou = IOU(box1, box2)
	
						if iou > self.nms_confidence:
							overlap_objects.append(obj_n2)
	
		last_object_list = []
	
		for remove_index in range(len(sorted_detections)):
			if remove_index not in overlap_objects:
				last_object_list.append(sorted_detections[remove_index][0])
	
		return list(set(last_object_list))

	def full_detection(self, frame, width, height):

		raw_detections = self.detect_objects(frame)
		selected_detections = self.select_detections(raw_detections, width, height)
		pro_detections = self.nms(selected_detections)

		return pro_detections