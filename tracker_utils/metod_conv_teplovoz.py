import pickle
import glob
import os
import math
import imutils
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision as tv
import math
import imutils
from abc import ABC, abstractmethod

class RailBoxManager():
    def __init__(self,is_plot = False):
        self.boxes = []
        self.is_plot = is_plot
        self.box_flip = None
        self.angle = []
        self.rotated_boxes = []
        self.conv = []
        self.rotation = None

    def upload(self, dir = ''):
        self.boxes = []
        paths = sorted(glob.glob(dir + '*.pkl'))
        for path in paths:
            with open(path, 'rb') as f:
                self.boxes.append(
                    torch.FloatTensor(
                        pickle.load(f)
                    ).transpose(2,0)
                )
        return self.boxes

    # Функция поворота и свёртки тензоров
    def angle_rotation_box(self, box, angle, box_flip):
        rotated_box = tv.transforms.functional.rotate(box_flip,angle)
        merged_boxes = torch.sum(torch.multiply(box,rotated_box))/torch.count_nonzero(rotated_box)
        # self.rotated_boxes.append(rotated_box)
        return merged_boxes
     # Функция
    def make_rotation_map(self, box):
        self.plot_tensor(box)
        # self.rotated_boxes = []
        step = 180/(box.shape[1]*2) # Шаг цикла
        self.angle = [x*step for x in range(box.shape[1]*2+1)]
        # bx = box[::2,::2]
        box_flip = tv.transforms.functional.vflip(box)
        conv = [float(self.angle_rotation_box(box, x, box_flip)) for x in self.angle]
        self.conv = torch.FloatTensor(conv)
        print(self.angle[int(torch.argmax(self.conv))]) # - в проекте вернуть это значение
        return self.conv

    def rotation_plot(self):
        plt.plot(self.angle,self.conv)
        plt.show()


    # def show_rotation(self):
    #     for tensor_box, angle in zip(self.rotated_boxes, self.angle):
    #         self.plot_tensor(tensor_box,angle)

    def plot_tensor(self,box_tensor,angle=0):
        if self.is_plot:
            box_tensor = box_tensor.transpose(2,0).numpy()
            plt.xlabel('angle = ' + str(angle))
            plt.imshow(box_tensor)
            plt.show()

# class Synogram(Loader):
#     def __init__(self):
#         self.rotation = None
#     def update(self, box, angle):
#         I = box.numpy()[::2,::2]
#         h, w = I.shape

#         t1 = time.time()
#         for _ in range(100):
#             t2 = time.time()
#             I = I - np.mean(I)  # Demean; make the brightness extend above and below zero
#             sinogram = radon(I, angle)
#             r = np.array([(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
#             self.rotation = np.argmax(r)
#             # print((time.time()-t2)*1000)

#         print((time.time()-t1)*1000)

#         print('self.rotation: {:.2f} degrees'.format(180-angle[self.rotation]))
#         print('self.rotation: {:.2f} num'.format(self.rotation))

manager  = RailBoxManager(True)
manager.upload('/home/user/dataset/boxes/')
conv = manager.make_rotation_map(manager.boxes[0])
manager.rotation_plot()
# manager.show_rotation()

import random
rand_indexes = [random.randint(0,len(manager.boxes))for _ in range(5)] 
for i in rand_indexes:
    conv = manager.make_rotation_map(manager.boxes[i])
