import requests 
import io
import cv2 
import numpy
import json

###################################################################

class FrameLimitError(Exception):
    pass

class SameFrameError(Exception):
    pass

class BadConnectionError(Exception):
    pass

class ClosedSessionError(Exception):
    pass

###################################################################

class Commcator:

	def __init__(self, username, password, connection_addr, frame_info_addr, send_addr, class_addr, base_url):

		self.username = username
		self.password = password

		self.base_url = base_url

		self.connection_addr = self.base_url+connection_addr
		self.frame_info_addr = self.base_url+frame_info_addr
		self.send_addr = self.base_url+send_addr
		self.class_addr = self.base_url+class_addr+"{}/"
	
		self.frame_info = []

		self.frame_id = ""
		self.frame_url = ""
		self.video_name = ""
		self.session = ""
		self.json_to_return = {}


	def init(self, start_frame):
		
		bilgi = {'username': self.username,
                   'password': self.password}

		response = requests.post(self.connection_addr, data=bilgi)
	
		status = response.status_code
		response_json = json.loads(response.text)

		detail = "no detail"

		try:
			detail = list(response_json.values())[0] if len(list(response_json.values())) > 0 else "no detail"
		except:
			pass

		if status == 200 or status == 201:
			self.token = response_json["token"]
			print("\033[92mAuthorization Successful: [Token: {}]".format(self.token))

			self.frame_info = self.get_frame_data()[start_frame:]
			print("\033[92mFrame Info Successfully Received: [Frame Count: {}]".format(len(self.frame_info)))

		elif status == 400:
			raise ClosedSessionError("Session Closed [Code: {}] [Details: {}]".format(status, detail))
		elif status == 403:
		    raise FrameLimitError("Frame Limit Exceeded [Code: {}] [Details: {}]".format(status, detail))
		elif status == 406:
		    raise SameFrameError("Frame Already Sent [Code: {}] [Details: {}]".format(status, detail))
		else:
		    raise BadConnectionError("Request Unsuccessful [Code: {}] [Details: {}]".format(status, detail))

	
	def get_frame_data(self):

		"""frame_info = []

		response = requests.get(self.frame_info_addr, headers={'Authorization': 'Token {}'.format(self.token)})

		status = response.status_code

		if status == 200 or status == 201:
			frame_info_raw = json.loads(response.text)
    	
			for frm in range(len(frame_info_raw)):
				frame_info.append(list(frame_info_raw[frm].values()))

		elif status == 400:
			raise ClosedSessionError("Session Closed [Code: {}]".format(status))
		elif status == 403:
		    raise FrameLimitError("Frame Limit Exceeded [Code: {}]".format(status))
		elif status == 406:
		    raise SameFrameError("Frame Already Sent [Code: {}]".format(status))
		else:
		    raise BadConnectionError("Request Unsuccessful [Code: {}]".format(status))

		return frame_info"""

		return [("https://i.hizliresim.com/", "https://i.hizliresim.com/9mcj019.jpg",
			 "test.mp4", "1") for x in range(200)]


	def set_frame_info(self, frame_n):

		self.frame_id, self.frame_url, self.video_name, self.session = self.frame_info[frame_n]


	def get_frame(self):

		headers = {'Authorization': 'Token {}'.format(self.token)}
    
		#get_image = requests.get(self.base_url+"media"+self.frame_url, stream=True, headers=headers, timeout=10)
		get_image = requests.get(self.frame_url, stream=True, timeout=5)
		status = get_image.status_code

		detail = "no detail"

		try:
			detail = list(get_image.values())[0] if len(list(get_image.values())) > 0 else "no detail"
		except:
			pass

		if status == 400:
			raise ClosedSessionError("Session Closed [Code: {}] [Details: {}]".format(status, detail))
		elif status == 403:
		    raise FrameLimitError("Frame Limit Exceeded [Code: {}] [Details: {}]".format(status, detail))
		elif status == 406:
		    raise SameFrameError("Frame Already Sent [Code: {}] [Details: {}]".format(status, detail))
		elif status != 200 and status != 201:
		    raise BadConnectionError("Request Unsuccessful [Code: {}] [Details: {}]".format(status, detail))

		img_stream = io.BytesIO(get_image.content)
		raw_frame = cv2.imdecode(numpy.frombuffer(img_stream.read(), numpy.uint8), 1)
		
		return raw_frame
	
	
	def create_json(self, object_list):

		json_list = []

		for object_n in range(len(object_list)):

			object_box = object_list[object_n]

			landing_status = object_box[1]
			class_id, x1, y1, x2, y2 = object_box[0]
		
			detection_json = {"cls": self.class_addr.format(class_id),
							  "landing_status": str(landing_status),
							  "top_left_x": str(x1),
							  "top_left_y": str(y1),
							  "bottom_right_x": str(x2),
							  "bottom_right_y": str(y2)}

			json_list.append(detection_json)

		self.json_to_return = {"frame": self.frame_id,
						  "detected_objects": json_list}

	
	def send_json(self, object_list):

		self.create_json(object_list)

		headers = {'Authorization': 'Token {}'.format(self.token), 
		'Content-Type': 'application/json',}
    
		send_json = requests.post(self.send_addr, json=self.json_to_return, headers=headers)
		status = send_json.status_code

		detail = "no detail"

		try:
			detail = list(send_json.values())[0] if len(list(send_json.values())) > 0 else "no detail"
		except:
			pass

		if status == 400:
			raise ClosedSessionError("Session Closed [Code: {}] [Details: {}]".format(status, detail))
		elif status == 403:
		    raise FrameLimitError("Frame Limit Exceeded [Code: {}] [Details: {}]".format(status, detail))
		elif status == 406:
		    raise SameFrameError("Frame Already Sent [Code: {}] [Details: {}]".format(status, detail))
		elif status != 200 and status != 201:
		    raise BadConnectionError("Request Unsuccessful [Code: {}] [Details: {}]".format(status, detail))