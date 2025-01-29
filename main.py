import sys
import warnings
import traceback

if not sys.warnoptions:
    warnings.simplefilter("ignore")

############################################################################################

import time
import os
import numpy

from modules.title_module import print_title
print_title()
from modules.prep_utils import remove_false_positive, set_landing_status
from modules.ai_module import Detector
from modules.comm_module import Commcator, FrameLimitError, SameFrameError
from modules.track_module import Tracker
from modules.log_module import Logger
from modules.constants import*

os.system('color')
if 1:
	print_title()

track_group_length = 5

image_path = "d:\\new_images"
json_path = "d:\\new_json"

try:
	os.mkdir(image_path)
except:
	pass

try:
	os.mkdir(json_path)
except:
	pass

######################## INITIALIZE LOGGER #######################

logger = Logger(0.01, True)
logger.start()

######################## INITIALIZE COMMUNICATOR #######################

start_frame = int(input("Input Start Frame >>> "))
os.system("cls")
print("Start Frame: {}\n\n".format(start_frame))
print()

print("\033[94mConnecting to Server...")
print()

while True:
	try:
		comm = Commcator("softwave", "QMSjLIOj", "auth/", "frames/", "prediction/", "classes/",
		"http://teknofest.cezerirobot.com:2052/")

		comm.init(start_frame)
		break
	except Exception:
		print("\033[91mError Occured:", traceback.format_exc())
		input()

print()
print("\033[92mConnection Successfully Established!")
print()

######################## INITIALIZE DETECTORS #######################

print("\033[94mDetectors Initializing...")
print()

while True:
	try:
		print("\033[94mCar Detector Initializing...")
		car_detector = Detector("weights\\car.weights", "weights\\car.cfg", 0.4, 0.4, (CAR_ID, ), 608)
		print("\033[92mCar Detector Successfully Initialized!")
		print()
		break
	except Exception:
		etype, value, tb = sys.exc_info()
		error_message = "".join(traceback.format_exception(etype, value, tb))
		print("\033[91mError Occured:", error_message)
		input()

while True:
	try:
		print("\033[94mPerson Detector Initializing...")
		person_detector = Detector("weights\\person.weights", "weights\\person.cfg", 0.5, 0.2, (PERSON_ID, ), 416)
		print("\033[92mPerson Detector Successfully Initialized!")
		print()
		break
	except Exception:
		etype, value, tb = sys.exc_info()
		error_message = "".join(traceback.format_exception(etype, value, tb))
		print("\033[91mError Occured:", error_message)
		input()

while True:
	try:
		print("\033[94mLanding Detector Initializing...")
		landing_detector = Detector("weights\\landing.weights", "weights\\landing.cfg", 0.4, 0.5, (LANDING_UAI_ID, LANDING_UAP_ID), 416)
		print("\033[92mLanding Detector Successfully Initialized!")
		print()
		break
	except Exception:
		etype, value, tb = sys.exc_info()
		error_message = "".join(traceback.format_exception(etype, value, tb))
		print("\033[91mError Occured:", error_message)
		input()

while True:
	try:
		print("\033[94mFalse Detectors Initializing...")
		false_detector = Detector("weights\\car.weights", "weights\\car.cfg", 0.4, 0.4, (1,2,3,4,5), 608)
		print("\033[92mFalse Detectors Successfully Initialized!")
		print()
		break
	except Exception:
		etype, value, tb = sys.exc_info()
		error_message = "".join(traceback.format_exception(etype, value, tb))
		print("\033[91mError Occured:", error_message)
		input()

print("\033[92mAll Detectors Successfully Initialized!")
print()

############################## MAINLOOP #############################

print("\033[94mGroup Count Calculating...")
group_count = int(int(len(comm.frame_info)/track_group_length) + len(comm.frame_info)%track_group_length)
print("\033[92mGroup Count Successfully Calculated! [Group Count: {}]".format(group_count))
print()

print("\033[92mAll Initialization Successful!")
print()

input("\033[39mPress Enter to Start >>> ")
print()

for group_n in range(group_count):

	logger.log("\n\nStarting Group: {}".format(group_n), 3)

	while True:
		try:
			image_folder = "{}\\{}".format(image_path, group_n)
			os.mkdir(image_folder)
			break
		except FileExistsError:
			logger.log("Empty the Image File!".format(group_n), 2)
			input()

	while True:
		try:
			json_folder = "{}\\{}".format(json_path, group_n)
			os.mkdir(json_folder)
			break
		except FileExistsError:
			logger.log("Empty the Json File!".format(group_n), 2)
			input()

	current_frame_list = []

	car_box_list = []
	person_box_list = []
	landing_box_list = []

	car_counter = 0
	landing_counter = 0
	person_counter = 0

	for order_n in range(track_group_length):

		tt1 = time.perf_counter()

		########################## GET FRAME ############################
		
		comm.set_frame_info(order_n+(group_n*track_group_length))
		logger.log("\nCurrent Frame => Net: {}+{} Order: {} Group: {} Name: {}\n".format(start_frame, order_n+(group_n*track_group_length), order_n, group_n, comm.frame_id), 4)

		while True:
			try:
				logger.log("Downloading Frame..", 0)
				t1 = time.perf_counter()

				raw_frame = comm.get_frame()

				t2 = time.perf_counter()
				logger.log("Frame Successfully Downloaded! [{}]\n".format(round(t2-t1, 3)), 1)
				break
			except FrameLimitError:
				logger.log("Frame Limit!", 3)
				time.sleep(0.1)
			except Exception:
				etype, value, tb = sys.exc_info()
				error_message = "".join(traceback.format_exception(etype, value, tb))
				logger.log("Error Occured: "+str(error_message), 2)
				input()

		pro_frame = raw_frame
		image_height, image_width = pro_frame.shape[:-1]
	
		########################## DETECTIONS ###########################
	
		logger.log("Car Detection Started...", 0)
		t1 = time.perf_counter()
		car_main_detections = car_detector.full_detection(pro_frame, 1920, 1080)
		t2 = time.perf_counter()
		logger.log("Car Detection Successful! [{}]\n".format(round(t2-t1, 3)), 1)

		logger.log("Person Detection Started...", 0)
		t1 = time.perf_counter()
		person_main_detections = person_detector.full_detection(pro_frame, 1920, 1080)
		t2 = time.perf_counter()
		logger.log("Person Detection Successful! [{}]\n".format(round(t2-t1, 3)), 1)

		logger.log("Landing Detection Started...", 0)
		t1 = time.perf_counter()
		landing_main_detections = landing_detector.full_detection(pro_frame, 1920, 1080)
		t2 = time.perf_counter()
		logger.log("Landing Detection Successful! [{}]\n".format(round(t2-t1, 3)), 1)

		logger.log("False Detection Started...", 0)
		t1 = time.perf_counter()
		false_detections = false_detector.full_detection(pro_frame, 1920, 1080)
		
		final_car_detections = remove_false_positive(car_main_detections, false_detections, 0.8)
		final_person_detections = remove_false_positive(person_main_detections, false_detections, 0.8)
		final_landing_detections = remove_false_positive(landing_main_detections, false_detections, 0.8)
		t2 = time.perf_counter()
		logger.log("False Detection Successful! [{}]\n".format(round(t2-t1, 3)), 1)

		logger.log("All Detections Successful!\n", 1)

		######################## SAVE FOR TRACKING #######################

		logger.log("Saving Started...", 0)

		current_frame_list.append(pro_frame)

		im_save_path = "{}\\{}.jpg".format(image_folder, order_n)
		cv2.imwrite(im_save_path, pro_frame)

		car_counter += len(final_car_detections)
		person_counter += len(final_person_detections)
		landing_counter += len(final_landing_detections)

		car_box_list.append(final_car_detections)
		person_box_list.append(final_person_detections)
		landing_box_list.append(final_landing_detections)

		logger.log("Saving Successful!\n", 1)

		tt2 = time.perf_counter()

		logger.log("Successful Frame => Net: {}+{} Order: {} Group: {} Name: {} [{}]".format(start_frame, order_n+(group_n*track_group_length), order_n, group_n, comm.frame_id, round(tt2-tt1, 3)), 5)



	########################### TRACK AND SEND ###########################

	logger.log("\n\nTracking Started... => Group: {} Car Count: {} Person Count: {} Landing Count: {} \n".format(group_n, car_counter, person_counter, landing_counter), 6)

	t1 = time.time()

	logger.log("Setting Trackers...", 0)
	
	car_tracker = Tracker(car_detector, false_detector)
	person_tracker = Tracker(person_detector, false_detector)
	landing_tracker = Tracker(landing_detector, false_detector)

	logger.log("Trackers Successfully Set!", 1)

	logger.log("Applying Interpolation and Tracking...", 0)

	if car_counter > 2:
		car_track_list = car_tracker.get_track_list(current_frame_list, car_box_list, 2000, numpy.sqrt(image_width**2 + image_height**2))
		new_car_interp = car_tracker.get_interp_list(car_track_list, car_box_list)
	else:
		new_car_interp = [[] for fn in range(len(car_box_list))]

	if person_counter > 2:
		person_track_list = person_tracker.get_track_list(current_frame_list, person_box_list, 2000, numpy.sqrt(image_width**2 + image_height**2))
		new_person_interp = person_tracker.get_interp_list(person_track_list, person_box_list)
	else:
		new_person_interp = [[] for fn in range(len(person_box_list))]

	if landing_counter > 2:
		landing_track_list = landing_tracker.get_track_list(current_frame_list, landing_box_list, 2000, numpy.sqrt(image_width**2 + image_height**2))
		new_landing_interp = landing_tracker.get_interp_list(landing_track_list, landing_box_list)
	else:
		new_landing_interp = [[] for fn in range(len(landing_box_list))]

	logger.log("Interpolation and Tracking Successfully Applied!", 1)

	t2 = time.time()

	logger.log("\nTracking Successful! => Group: {} [{}]\n".format(group_n, t2-t1), 5)
	
	for order_n in range(track_group_length):

		logger.log("Improving the Frame...", 0)
		
		current_box_list = car_box_list[order_n] + person_box_list[order_n] + landing_box_list[order_n]
	
		enhanced_detections = current_box_list+new_car_interp[order_n]+new_person_interp[order_n]+new_landing_interp[order_n]

		##################### CREATE JSON AND SEND #######################

		enhanced_detections_with_landing = set_landing_status(enhanced_detections)
		comm.set_frame_info(order_n+(group_n*track_group_length))

		logger.log("Frame Successfully Improved!", 1)

		logger.log("\nSending Prediction... => Net: {}+{} Order: {} Group: {} Name: {}".format(start_frame, order_n+(group_n*track_group_length), order_n, group_n, comm.frame_id), 7)

		t1 = time.time()

		while True:
			try:
				comm.send_json(enhanced_detections_with_landing)

				js_save_path = "{}\\{}.txt".format(json_folder, order_n)
				jsf = open(js_save_path, "w")
				jsf.write(str(comm.json_to_return))
				jsf.close()

				break
			except FrameLimitError:
				logger.log("Frame Limit!", 2)
				time.sleep(0.1)
			except SameFrameError:
				logger.log("Same Frame!", 2)
				break
			except Exception as ex_details:
				etype, value, tb = sys.exc_info()
				error_message = "".join(traceback.format_exception(etype, value, tb))
				logger.log("Error Occured: "+str(error_message), 2)
				input()

		t2 = time.time()

		logger.log("\nPrediction Successfully Sent! => Net: {}+{} Order: {} Group: {} Name: {} [{}]".format(start_frame, order_n+(group_n*track_group_length), order_n, group_n, comm.frame_id, t2-t1), 5)
