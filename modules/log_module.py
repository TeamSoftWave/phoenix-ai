import time
from threading import Thread
from colorama import Fore, Back, init


class Logger:
	def __init__(self, delay, log_cond):

		init()

		self.log_cond = log_cond

		if self.log_cond:
			self.log_file = open("log.txt", "w")
			self.log_file.close()

		self.delay = delay
		self.log_queue = []
		self.log_loop_thread = Thread(target=self.log_loop)

		self.colors = [Fore.LIGHTBLUE_EX, Fore.GREEN, Fore.LIGHTRED_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTGREEN_EX, Fore.WHITE, Fore.YELLOW]
		#def_apply, def_success, def_error, group_start, frame_start, frame_succes, track_start, prediction_start 

	def start(self):
		self.loop_cond = True
		self.log_loop_thread.start()

	def stop(self):
		self.loop_cond = False

	def log(self, message, color):
		self.log_queue.append((message, self.colors[color]))

	def log_loop(self):
		while self.loop_cond:
			if len(self.log_queue) > 0:
				print(self.log_queue[0][1]+self.log_queue[0][0])

				if self.log_cond:
					self.log_file = open("log.txt", "a")
					self.log_file.write(self.log_queue[0][0])
					self.log_file.close()

				self.log_queue.pop(0)
				time.sleep(self.delay)