import numpy as np
import albumentations as alb
import cv2
import random
import math
import glob
import numpy
import shutil
import os

class Processings:
    def __init__(self):
        self.channel_shuffle = alb.Compose([alb.augmentations.transforms.ChannelShuffle(always_apply=1, p=0.8)])

    def rotating_img(self, img, angle):
        return img

    def channel_shuffle(self, img):
        img = self.channel_shuffle(image=img)
        return img

