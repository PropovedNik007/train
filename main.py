from pprint import pprint
import numpy as np
import cv2
import os
import pickle
import random
import sys
from iou import *
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

def data_center_x(filtered_data):
    cx = filtered_data[:, 0] + (filtered_data[:, 2] - filtered_data[:, 0]) / 2
    return cx


def analytics(input_data, output_image):

    data = np.array(input_data, dtype=int)

    class_rails = data[:, 4]
    class_rails_filter = np.where((class_rails == 0) | (class_rails == 3) | (class_rails >= 8))
    filtered_data = data[class_rails_filter]

    if Memory.previous_data == []:
        Memory.previous_data = np.array(filtered_data, dtype=int)

    if (np.std(filtered_data[-5:, 2]) <= 30):
        a = filtered_data[(-len(filtered_data[-5:, 2])) - (np.where(np.std(filtered_data[-5:, 2]) <= 30)[0])]
        Memory.previous_data[-5:, 2] = filtered_data[-5:, 2]
    else:
        a = Memory.previous_data[(-len(Memory.previous_data[-5:, 2])) - (np.where(np.std(Memory.previous_data[-5:, 2]) <= 30)[0])]

    if (np.std(filtered_data[0:5, 2]) <= 30):
        b = filtered_data[np.where(np.std(filtered_data[0:5, 2]) <= 30)[0]]
        Memory.previous_data[0:5, 2] = filtered_data[0:5, 2]
    else:
        b = Memory.previous_data[np.where(np.std(Memory.previous_data[0:5, 2]) <= 50)[-1]]

    y = filtered_data[:, 3]
    # cv2.line(output_image, (a[0][0], a[0][3]), (b[0][0], b[0][3]), (255, 0, 255), thickness=3)
    # cv2.line(output_image, (a[0][2], a[0][3]), (b[0][2], b[0][3]), (255, 0, 255), thickness=3)

    x1 = (b[:, 0] * a[:, 1] - y[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
    x2 = (b[:, 2] * a[:, 3] - y[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])

    cv2.line(output_image, (int(x1[0]), y[0]), (int(x1[-1]), y[-1]), (255, 0, 255), thickness=3)
    cv2.line(output_image, (int(x2[0]), y[0]), (int(x2[-1]), y[-1]), (255, 0, 255), thickness=3)

    cx = data_center_x(filtered_data)
    filtered_data = filtered_data[np.where((cx <= x2) & (cx >= x1))]
    # filtered_data = filtered_data[np.where(cx in range(x1, x2))]

    # for i in range(len(x1)):
    #     cv2.circle(output_image, (int(x1[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    #     cv2.circle(output_image, (int(x2[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    #     cv2.circle(output_image, (int(cx[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    class_rails = filtered_data[:, 4]
    if any(class_rails == box_class.index("switch right front")):
        Memory.switch_right_front += 1
        # print("right", Memory.switch_right_front)

    elif any(class_rails == box_class.index('switch left front')):
        Memory.switch_left_front += 1
        # print("left", Memory.switch_left_front)
    if (Memory.init_box_detected == []) & (len(filtered_data) == len(Memory.y_lines)):
        Memory.init_box_detected = filtered_data
        lines_data = Memory.init_box_detected
    else:
        lines_data = filtered_data
    for i, line in enumerate(Memory.y_lines):
        y_boxes = filtered_data[np.where(filtered_data[i, 3] == line[3])]
        overlap = []
        if len(y_boxes) > 1:
            for j, box in enumerate(y_boxes):
                ################
                # тут продолжить
                #################
                overlap.append(area(y_boxes[j], filtered_data[i]))
                lines_data[i] = y_boxes[np.where(overlap == max(overlap))[0][0]]
        else:
            lines_data[i] = y_boxes[0]
        # lines_data[i] = filtered_data[np.where(filtered_data[i, 3] == line)]

        class_rails = lines_data[i][4]

        if class_rails == box_class.index('double central rails'):
            cx_right = lines_data[i][2] - Memory.previous_data[i, 2] + Memory.previous_data[i, 0]
            cx_left = lines_data[i][0] + Memory.previous_data[i, 2] - Memory.previous_data[i, 0]

            # if cx1
            # y_double_central_rails = line[1]
            # x1_double_central_rails = (b[:, 0] * a[:, 1] - y_double_central_rails[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
            # x2_double_central_rails = (b[:, 2] * a[:, 3] - y_double_central_rails[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])
            if Memory.switch_left_front >= 20:
                filtered_data[i, 0] = int(cx_left)
                # double_central_rails[:, 0] = cx_left[:]
                print("LEFT")
                # filtered_data = double_central_rails[np.where(~(cx_right <= x2_double_central_rails) & ~(cx_right >= x1_double_central_rails))]
            if Memory.switch_right_front >= 20:
                filtered_data[i, 0] = int(cx_right)
                # cv2.circle(output_image,
                #            (int(filtered_data[double_central_rails_filter][:][0][0]), y_double_central_rails), 40,
                #            (0, 255, 0), thickness=-1)
                print("RIGHT")
                # filtered_data[double_central_rails[np.where(~(cx_right <= x2_double_central_rails) & ~(cx_right >= x1_double_central_rails), 1, 0)]] = [x1_double_central_rails, ...]

    # print("left", Memory.switch_left_front)
        #
        # elif class_rails == box_class.index('right crossing') and Memory.switch_right_front >= 20:
        #
        #
        # elif class_rails == box_class.index('left crossing') and Memory.switch_left_front >= 20:
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


    if (np.std(filtered_data[-5:, 2]) <= 30):
        a = filtered_data[(-len(filtered_data[-5:, 2])) - (np.where(np.std(filtered_data[-5:, 2]) <= 30)[0])]
        Memory.previous_data[-5:, 2] = filtered_data[-5:, 2]
    else:
        a = Memory.previous_data[(-len(Memory.previous_data[-5:, 2])) - (np.where(np.std(Memory.previous_data[-5:, 2]) <= 30)[0])]

    if (np.std(filtered_data[0:5, 2]) <= 30):
        b = filtered_data[np.where(np.std(filtered_data[0:5, 2]) <= 30)[0]]
        Memory.previous_data[0:5, 2] = filtered_data[0:5, 2]
    else:
        b = Memory.previous_data[np.where(np.std(Memory.previous_data[0:5, 2]) <= 50)[-1]]

    y = filtered_data[:, 3]
    # cv2.line(output_image, (a[0][0], a[0][3]), (b[0][0], b[0][3]), (255, 0, 255), thickness=3)
    # cv2.line(output_image, (a[0][2], a[0][3]), (b[0][2], b[0][3]), (255, 0, 255), thickness=3)

    x1 = (b[:, 0] * a[:, 1] - y[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
    x2 = (b[:, 2] * a[:, 3] - y[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])

    cv2.line(output_image, (int(x1[0]), y[0]), (int(x1[-1]), y[-1]), (255, 0, 255), thickness=3)
    cv2.line(output_image, (int(x2[0]), y[0]), (int(x2[-1]), y[-1]), (255, 0, 255), thickness=3)

        # cx = data_center_x(filtered_data)
        # filtered_data = filtered_data[np.where((cx <= x2) & (cx >= x1))]
        # if any(class_rails == box_class.index('double central rails')):
        #     double_central_rails_filter = class_rails == box_class.index('double central rails')
        #     double_central_rails = filtered_data[double_central_rails_filter]
        #     cx_right = data_center_x(double_central_rails) + ((double_central_rails[:, 2] - double_central_rails[:, 0]) / 4)
        #     cx_left = data_center_x(double_central_rails) - ((double_central_rails[:, 2] - double_central_rails[:, 0]) / 4)
        #
        #     # if cx1
        #     y_double_central_rails = double_central_rails[:, 1]
        #     x1_double_central_rails = (b[:, 0] * a[:, 1] - y_double_central_rails[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
        #     x2_double_central_rails = (b[:, 2] * a[:, 3] - y_double_central_rails[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])
        #     if Memory.switch_left_front >= 20:
        #         pass
        #         # filtered_data = double_central_rails[np.where(~(cx_right <= x2_double_central_rails) & ~(cx_right >= x1_double_central_rails))]
        #     if Memory.switch_right_front >= 20:
        #         pass
        #         # filtered_data[double_central_rails[np.where(~(cx_right <= x2_double_central_rails) & ~(cx_right >= x1_double_central_rails), 1, 0)]] = [x1_double_central_rails, ...]
        # filtered_data = filtered_data[np.where((cx1 <= x2) & (cx1 >= x1))]
        # filtered_data = filtered_data[np.where((cx2 <= x2) & (cx2 >= x1))]
        #
        # if cx1 in range(int(Memory.init_box_detected[0]), int(Memory.init_box_detected[2])):
        #     cv2.circle(output_image, (cx1, cy), 40, (0, 255, 0), thickness=-1)
        #     Memory.current_frame_rails.append(box)
        #     Memory.previous_box[0] = self.box_center_x(box)
        #     Memory.init_box_detected = box
        #     Memory.init_box_detected[0] = self.box_center_x(box)
        #
        # else:
        #     cv2.circle(output_image, (cx2, cy), 40, (0, 255, 0), thickness=-1)
        #     Memory.current_frame_rails.append(box)
        #     Memory.init_box_detected = box
        #     Memory.previous_box[2] = self.box_center_x(box)


    # output_data = class_rails_filter
    output_data = filtered_data

    return output_data
