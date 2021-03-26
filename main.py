from pprint import pprint
import numpy as np
import cv2
import os
import pickle
import random
import sys
from memory import *

box_class = ["double central rails", "double left rails", "double right rails", "central rails", "left rails",
            "right rails", "half left rails", "half right rails", "switch right back", "switch left back", "switch right front",
            "switch left front", "switch", "left crossing", "right crossing", "double cross"]

def box_center_x(box):
    cx = int(box[0]) + int((box[2] - box[0]) / 2)
    return cx

def box_center_y(box):
    cy = box[1] + int((box[3] - box[1]) / 2)
    return cy

def analytics(input_data, output_image):
    data = np.array(input_data, dtype=int)
    class_rails = data[:, 4]
    class_rails_filter = np.where((class_rails == 0) | (class_rails == 3) | (class_rails >= 8))
    filtered_data = data[class_rails_filter]

    central_rails = filtered_data[np.where(filtered_data[:, 4] == 3)]
    # print(central_rails)
    # b = central_rails[np.where(central_rails[:, 3] == np.min(central_rails[:, 3]))]
    # b = filtered_data[np.where(abs(filtered_data[:, 2] - np.mean(filtered_data[0:5, 2])) <= 10)]
    # a = filtered_data[np.where(abs(filtered_data[:, 2] - np.mean(filtered_data[-5:, 2])) <= 10)]

    # try:
    # a = filtered_data[np.where(np.std(filtered_data[-5:, 2]) <= 100)]
    a = filtered_data[(-len(filtered_data[-5:, 2])) - (np.where(np.std(filtered_data[-10:, 2]) <= 100)[0])]
    b = filtered_data[np.where(np.std(filtered_data[0:10, 2]) <= 100)]
    # print(np.std(filtered_data[0:5, 2]))
    # print(np.std(filtered_data[-5:, 2]))
    by = b[np.where(b[:, 3] == np.min(b[:, 3]))]
    ay = a[np.where(a[:, 3] == np.max(a[:, 3]))]

    axy = ay[np.where(ay[:, 2] == np.min(ay[:, 2]))]
    bxy = by[np.where(by[:, 2] == np.max(by[:, 2]))]
    print("a", axy)
    print("b", bxy)
    # except Exception as e:
    #     print(e)
    #     sys.exit(0)

    x = filtered_data[:, 2]
    y = filtered_data[:, 3]
    cv2.line(output_image, (axy[0][2], axy[0][3]), (bxy[0][2], bxy[0][3]), (255, 0, 255), thickness=3)

    # vector = central_rails[np.where((y - a[3]) / (b[3] - a[3]) == (x - a[2]) / (b[2] - a[2]))]
    # vector = central_rails[np.where(abs(y - a[3]) / (b[3] - a[3]) - (x - a[2]) / (b[2] - a[2])) <= 100]
    # vector = central_rails[np.where((abs(y - a[3]) / (b[3] - a[3]) - (x - a[2]) / (b[2] - a[2])) <= 100)]
    # (y[:] - a[:, 3]) / (b[:, 3] - a[:, 3]) - (x[:] - a[:, 2]) / (b[:, 2] - a[:, 2])

    # cv2.imshow('window_namedd', output_image)
    # cv2.waitKey(0)

    #
    # if class_rails == box_class.index("switch right front") and Memory.between_frame_switch_front == 0:
    #     Memory.previous_switch_right_front += 1
    #     Memory.between_frame_switch_front += 1
    #     Memory.between_frame_counter += 1
    #
    # elif class_rails == box_class.index('switch right front') and Memory.between_frame_switch_front == 0:
    #     Memory.previous_switch_right_front += 1
    #     Memory.between_frame_switch_front += 1
    #     Memory.between_frame_counter += 1
    #
    # elif class_rails == box_class.index('right crossing') and Memory.previous_switch_right_front >= 10:
    #     pass
    #
    # elif class_rails == box_class.index('switch left front'):
    #     Memory.previous_switch_left_front += 1
    #
    # elif class_rails == box_class.index('left crossing') and Memory.previous_switch_left_front >= 10:
    #     pass
    #
    # elif class_rails == box_class.index('central rails'):
    #     Memory.previous_switch_left_front = 0
    #     Memory.previous_switch_right_front = 0
    #
    # elif class_rails == box_class.index('switch'):
    #     # Memory.previous_switch_left_front = 0
    #     # Memory.previous_switch_right_front = 0
    #     pass
    #
    # elif class_rails == box_class.index('double central rails'):
    #     pass
    #     # cx1 = box_center_x(data[:, 0]) + int((data[:, 2] - data[:, 0]) / 4)
    #     # cx2 = box_center_x(box) - int((box[2] - box[0]) / 4)
    #     # cy = box_center_y(box)
    #     #
    #     # if cx1 in range(int(Memory.init_box_detected[0]), int(Memory.init_box_detected[2])):
    #     #     cv2.circle(output_image, (cx1, cy), 40, (0, 255, 0), thickness=-1)
    #     #     Memory.current_frame_rails.append(box)
    #     #     Memory.previous_box[0] = self.box_center_x(box)
    #     #     Memory.init_box_detected = box
    #     #     Memory.init_box_detected[0] = self.box_center_x(box)
    #     #
    #     # else:
    #     #     cv2.circle(output_image, (cx2, cy), 40, (0, 255, 0), thickness=-1)
    #     #     Memory.current_frame_rails.append(box)
    #     #     Memory.init_box_detected = box
    #     #     Memory.previous_box[2] = self.box_center_x(box)
    #

    # output_data = class_rails_filter
    output_data = filtered_data
    # output_data = np.delete(data, -1, )

    # print(output_data)
    # print('00', output_data)

    return output_data
