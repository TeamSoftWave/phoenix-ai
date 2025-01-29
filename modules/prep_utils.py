import numpy
import cv2
from modules.constants import PERSON_ID, CAR_ID, LANDING_UAI_ID, LANDING_UAP_ID 


def IOU(box1, box2):

	inter_x = min(max(box1[1], box1[3]), max(box2[1], box2[3])) - max(min(box1[1], box1[3]), min(box2[1], box2[3]))
	inter_y = min(max(box1[2], box1[4]), max(box2[2], box2[4])) - max(min(box1[2], box1[4]), min(box2[2], box2[4]))

	if (inter_x < 0) or (inter_y < 0):
		inter_area = 0
	else:
		inter_area = inter_x*inter_y


	box1_area = abs(box1[1]-box1[3])*abs(box1[2]-box1[4])
	box2_area = abs(box2[1]-box2[3])*abs(box2[2]-box2[4])
	union_area = box1_area+box2_area-inter_area

	return inter_area/union_area


def is_intersecting(box1, box_list):

	intersection = False

	for box2 in box_list:

		if box1 != box2:
			if (box1[1] > box2[3] or box1[3] < box2[1] or box1[2] > box2[4] or box1[4] < box2[2]) == False:
				intersection = True
				break

	return intersection


def remove_false_positive(main_list, false_list, treshold):

	false_in_main = []

	for false_n in range(len(false_list)):

		max_iou = 0
		max_iou_index = 0

		for main_n in range(len(main_list)):

			new_iou = IOU(main_list[main_n], false_list[false_n])

			if new_iou > max_iou:
				max_iou_index = main_n
				max_iou = new_iou

		if max_iou >= treshold:
			false_in_main.append(max_iou_index)


	return [main_list[n] for n in range(len(main_list)) if n not in false_in_main]


def set_landing_status(detection_list):

	detection_list_with_landing = []

	for detection_n in range(len(detection_list)):
		detection = detection_list[detection_n]

		if detection[0] not in (LANDING_UAI_ID, LANDING_UAP_ID):
			detection_list_with_landing.append((detection_list[detection_n], -1))
		elif is_intersecting(detection_list[detection_n], detection_list):
			detection_list_with_landing.append((detection_list[detection_n], 0))
		else:
			detection_list_with_landing.append((detection_list[detection_n], 1))

	return detection_list_with_landing

