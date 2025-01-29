import numpy, cv2, copy
from modules.prep_utils import IOU, remove_false_positive
from modules.proc_module import Processings
from scipy import interpolate


class Tracker:

	def __init__(self, main_detector, false_detector):
		self.main_detector = main_detector 
		self.false_detector = false_detector

	def is_out_frame(self, box, width, height):

		x1 = numpy.min(box[1], box[3])
		x2 = numpy.max(box[1], box[3])
		y1 = numpy.min(box[2], box[4])
		y2 = numpy.max(box[2], box[4])

		return x1 > 0 and x2 < width and y1 > 0 and y2 < height

	def get_accordance(self, img1, img2):
		img1 = cv2.resize(img1, (40, 40))
		img2 = cv2.resize(img2, (40, 40))
		
		img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		
		sift = cv2.SIFT_create()
		
		kp1, des1 = sift.detectAndCompute(img1_gray, None)
		kp2, des2 = sift.detectAndCompute(img2_gray, None)
		
		index_params = dict(algorithm = 1, trees = 5)
		search_params = dict(checks=50)

		feature_accordance = 0

		try:
			flann = cv2.FlannBasedMatcher(index_params,search_params)
			matches = flann.knnMatch(des1,des2,k=2)

			feature_accordance = len([x[0] for x in matches if x[0].distance < 0.7*x[1].distance])
		except:
			pass

		
		blue_hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
		green_hist1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
		red_hist1 = cv2.calcHist([img1], [2], None, [256], [0, 256])
		
		blue_hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
		green_hist2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
		red_hist2 = cv2.calcHist([img2], [2], None, [256], [0, 256])
		
		blue_mean1 = numpy.sum([blue_hist1[x]*x for x in range(len(blue_hist1))])/numpy.sum(blue_hist1)
		green_mean1 = numpy.sum([green_hist1[x]*x for x in range(len(green_hist1))])/numpy.sum(green_hist1)
		red_mean1 = numpy.sum([red_hist1[x]*x for x in range(len(red_hist1))])/numpy.sum(red_hist1)
		
		blue_mean2 = numpy.sum([blue_hist2[x]*x for x in range(len(blue_hist2))])/numpy.sum(blue_hist2)
		green_mean2 = numpy.sum([green_hist2[x]*x for x in range(len(green_hist2))])/numpy.sum(green_hist2)
		red_mean2 = numpy.sum([red_hist2[x]*x for x in range(len(red_hist2))])/numpy.sum(red_hist2)
		
		blue_diff = abs(blue_mean1-blue_mean2)
		green_diff = abs(green_mean1-green_mean2)
		red_diff = abs(red_mean1-red_mean2)
		
		color_accordance = ((max(blue_mean1, blue_mean2)
						    +max(green_mean1, green_mean2)
						    +max(red_mean1, red_mean2))
						    -(blue_diff+red_diff+green_diff))

		return feature_accordance*color_accordance
	
	def get_track_list(self, frame_list, main_box_list, threshold, max_distance):
	
		track_list = [[(0, box)] for box in main_box_list[0]]
		
		for frame_n in range(1, len(frame_list)):
	
			new_box_list = main_box_list[frame_n]
			new_image = frame_list[frame_n]

			accordance_list = [[] for n in range(len(new_box_list))]

			for new_box_n in range(len(new_box_list)):

				new_box = new_box_list[new_box_n]

				y1, y2 = numpy.sort(numpy.array((int(new_box[2]), int(new_box[4]))).clip(min=0))
				x1, x2 = numpy.sort(numpy.array((int(new_box[1]), int(new_box[3]))).clip(min=0))

				new_center = (x2-x1, y2-y1)
				new_box_image = new_image[y1:y2, x1:x2]

				for serie_n in range(len(track_list)): 

					old_box = track_list[serie_n][-1][1]
					old_image = frame_list[track_list[serie_n][-1][0]]

					if old_box[0] == new_box[0]:

						oy1, oy2 = numpy.sort(numpy.array((int(old_box[2]), int(old_box[4]))).clip(min=0))
						ox1, ox2 = numpy.sort(numpy.array((int(old_box[1]), int(old_box[3]))).clip(min=0))

						old_center = (ox2-ox1, oy2-oy1)
						center_distance = numpy.sqrt((old_center[0]-new_center[0])**2 + (old_center[1]-new_center[1])**2)
						distance_factor = 1-(center_distance/max_distance)

						old_box_image = old_image[oy1:oy2, ox1:ox2]
	
						accordance_list[new_box_n].append(self.get_accordance(old_box_image, new_box_image)*distance_factor)
					else:
						accordance_list[new_box_n].append(0)

			accordance_array = numpy.array(accordance_list)

			selected_index = -1

			for new_box_n in range(len(accordance_array)):

				new_box = new_box_list[new_box_n]

				single_accordance_array = accordance_array[new_box_n]
				success_index_sorted = numpy.argsort(single_accordance_array)[::-1]

				success_check_array = [success_n for success_n in success_index_sorted if (numpy.argmax(accordance_array[:, success_n]) == new_box_n 
																	 and single_accordance_array[success_n] > threshold)]
				if len(success_check_array) > 0:
					track_list[copy.deepcopy(success_check_array[0])].append((frame_n, new_box))
				else:
					track_list.append([(frame_n, new_box)])
					
		return track_list

	def get_interp_list(self, track_list, box_list):

		new_boxes = [[] for fn in range(len(box_list))]

		for obj in track_list:

			frame_x1 = []
			frame_y1 = []
			frame_x2 = []
			frame_y2 = []
			must_interp = []
			frames = []
			i = 0

			for obj_frame in obj:

				class_id = obj_frame[1][0]
				x1 = obj_frame[1][1]
				y1 = obj_frame[1][2]
				x2 = obj_frame[1][3]
				y2 = obj_frame[1][4]
				frame_x1.append(x1)
				frame_y1.append(y1)
				frame_x2.append(x2)
				frame_y2.append(y2)

				if (obj_frame != obj[0]):
					if i != obj_frame[0]:
						diff = abs(obj_frame[0] - i)
						for x in range(diff):
							must_interp.append(i + x)
						i = obj_frame[0] + diff
					else: i += 1
				else: i += 1

				frames.append(obj_frame[0])

			frames = numpy.array(frames)
			frame_x1 = numpy.array(frame_x1)
			frame_y1 = numpy.array(frame_y1)
			frame_x2 = numpy.array(frame_x2)
			frame_y2 = numpy.array(frame_y2)

			if len(frame_x1) > 1:
				interp_x1 = interpolate.interp1d(frames, frame_x1)
				interp_y1 = interpolate.interp1d(frames, frame_y1)
				interp_x2 = interpolate.interp1d(frames, frame_x2)
				interp_y2 = interpolate.interp1d(frames, frame_y2)

				for frame in must_interp:
					interpolated_box = (class_id, float(interp_x1(frame)), float(interp_y1(frame)), float(interp_x2(frame)), float(interp_y2(frame)))
					current_box_list = box_list[frame]

					for current_box in current_box_list:
						if IOU(interpolated_box[1], current_box) >= 0.7 and class_id == current_box[0]:
							break
						elif current_box == current_box_list[-1]:
							new_boxes[int(frame)].append(interpolated_box)

		return new_boxes