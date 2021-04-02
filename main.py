from pprint import pprint
import numpy as np
import cv2
from memory import *
import scipy.ndimage.filters as filt
from scipy.ndimage.filters import gaussian_filter

box_class = ["double central rails", "double left rails", "double right rails", "central rails", "left rails",
             "right rails", "half left rails", "half right rails", "switch right back", "switch left back",
             "switch right front",
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

    # if Memory.previous_data == []:
    #     Memory.previous_data = np.array(filtered_data, dtype=int)

    if (np.std(filtered_data[-5:, 2]) <= 30):
        a = filtered_data[(-len(filtered_data[-5:, 2])) - (np.where(np.std(filtered_data[-5:, 2]) <= 30)[0])]
        Memory.previous_data[-5:, 2] = filtered_data[-5:, 2]
    else:
        a = Memory.previous_data[
            (-len(Memory.previous_data[-5:, 2])) - (np.where(np.std(Memory.previous_data[-5:, 2]) <= 30)[0])]

    if (np.std(filtered_data[0:5, 2]) <= 30):
        b = filtered_data[np.where(np.std(filtered_data[0:5, 2]) <= 30)[0]]
        Memory.previous_data[0:5, 2] = filtered_data[0:5, 2]
    else:
        b = Memory.previous_data[np.where(np.std(Memory.previous_data[0:5, 2]) <= 50)[-1]]

    y = filtered_data[:, 3]

    x1 = (b[:, 0] * a[:, 1] - y[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
    x2 = (b[:, 2] * a[:, 3] - y[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])
    #
    # cv2.line(output_image, (int(x1[0]), y[0]), (int(x1[-1]), y[-1]), (255, 0, 255), thickness=3)
    # cv2.line(output_image, (int(x2[0]), y[0]), (int(x2[-1]), y[-1]), (255, 0, 255), thickness=3)

    cx = data_center_x(filtered_data)
    filtered_data = filtered_data[np.where((cx <= x2) & (cx >= x1))]

    central_rails = filtered_data[np.where(filtered_data[:, 4] == 3)]
    if (Memory.x1_perspective == []):
        if (len(central_rails) >= 5):
            a_perspective = central_rails[0]
            b_perspective = central_rails[-1]
            # perspective = central_rails

            Memory.x1_perspective = (b_perspective[0] * a_perspective[1] - y[:] * (
                        b_perspective[0] - a_perspective[0]) -
                                     a_perspective[0] * b_perspective[1]) / (a_perspective[1] - b_perspective[1])
            Memory.x2_perspective = (b_perspective[2] * a_perspective[3] - y[:] * (
                        b_perspective[2] - a_perspective[2]) - a_perspective[2] * b_perspective[3]) / (
                                                a_perspective[3] - b_perspective[3])
    else:
        pass
        # cv2.line(output_image, (int(Memory.x1_perspective[0]), y[0]), (int(Memory.x1_perspective[-1]), y[-1]),
        #          (255, 255, 255), thickness=3)
        # cv2.line(output_image, (int(Memory.x2_perspective[0]), y[0]), (int(Memory.x2_perspective[-1]), y[-1]),
        #          (255, 255, 255), thickness=3)
        # print("x1", Memory.x1_perspective)
        # print("x2", Memory.x2_perspective)

    # filtered_data = filtered_data[np.where(cx in range(x1, x2))]

    # for i in range(len(x1)):
    #     cv2.circle(output_image, (int(x1[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    #     cv2.circle(output_image, (int(x2[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    #     cv2.circle(output_image, (int(cx[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)

    class_rails = filtered_data[:, 4]
    if any(class_rails == box_class.index("switch right front")):
        Memory.switch_right_front += 1
        print("right", Memory.switch_right_front)

    elif any(class_rails == box_class.index("switch left front")):
        Memory.switch_left_front += 1
        print("right", Memory.switch_left_front)

    elif any(class_rails == box_class.index("right crossing")):
        Memory.right_crossing += 1
        print("right crossing", Memory.right_crossing)

    elif any(class_rails == box_class.index("left crossing")):
        Memory.left_crossing += 1
        print("right crossing", Memory.left_crossing)

    if (Memory.switch_right_front >= 20) | (Memory.right_crossing >= 20):
        right_double_box = filtered_data[filtered_data[:, 4] != 17]
        filtered_data = right_double_box
        print("right", Memory.right_crossing, Memory.switch_right_front)

    if (Memory.switch_left_front >= 20) | (Memory.left_crossing >= 20):
        left_double_box = filtered_data[filtered_data[:, 4] != 16]
        filtered_data = left_double_box
        print("left", Memory.left_crossing, Memory.switch_left_front)

    if (np.std(filtered_data[-5:, 2]) <= 30):
        a = filtered_data[(-len(filtered_data[-5:, 2])) - (np.where(np.std(filtered_data[-5:, 2]) <= 30)[0])]
        Memory.previous_data[-5:, 2] = filtered_data[-5:, 2]
    else:
        a = Memory.previous_data[
            (-len(Memory.previous_data[-5:, 2])) - (np.where(np.std(Memory.previous_data[-5:, 2]) <= 30)[0])]

    if (np.std(filtered_data[0:5, 2]) <= 30):
        b = filtered_data[np.where(np.std(filtered_data[0:5, 2]) <= 30)[0]]
        Memory.previous_data[0:5, 2] = filtered_data[0:5, 2]
    else:
        b = Memory.previous_data[np.where(np.std(Memory.previous_data[0:5, 2]) <= 50)[-1]]

    y = filtered_data[:, 3]

    x1 = (b[:, 0] * a[:, 1] - y[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
    x2 = (b[:, 2] * a[:, 3] - y[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])
    cx = data_center_x(filtered_data)
    filtered_data = filtered_data[np.where((cx <= x2) & (cx >= x1))]

    # cv2.line(output_image, (int(x1[0]), y[0]), (int(x1[-1]), y[-1]), (255, 0, 255), thickness=3)
    # cv2.line(output_image, (int(x2[0]), y[0]), (int(x2[-1]), y[-1]), (255, 0, 255), thickness=3)

    # output_data = class_rails_filter
    output_data = filtered_data

    # polygon = (np.column_stack((filtered_data[:, 2], filtered_data[:, 3])), np.dtype('int32'))
    # polygon = (np.column_stack((filtered_data[:, 2], filtered_data[:, 3])), np.dtype('int32'))
    polygon_left = (np.column_stack((filtered_data[:, 0], filtered_data[:, 3])))
    polygon_right = (np.column_stack((filtered_data[::-1, 2], filtered_data[::-1, 3])))
    polygon = (np.row_stack((polygon_left, polygon_right)), np.dtype('int32'))[0]

    for i, box in enumerate(polygon):
        cv2.line(output_image, (int(polygon[i][0]), int(polygon[i][1])), (int(polygon[i-1][0]), int(polygon[i-1][1])), (255, 0, 255), thickness=3)
    filtered_data[:, 0] = filt.uniform_filter(filtered_data[:, 0], size=5, mode='nearest')
    # filtered_data[:, 0] = gaussian_filter(filtered_data[:, 0], sigma=(5), mode='nearest')
    filtered_data[:, 3] = filt.uniform_filter(filtered_data[:, 3], size=5, mode='nearest')
    filtered_data[::-1, 2] = filt.uniform_filter(filtered_data[::-1, 2], size=5, mode='nearest')
    # filtered_data[::-1, 2] = gaussian_filter(filtered_data[::-1, 2], sigma=(5), mode='nearest')
    # filtered_data[::-1, 3] = filt.uniform_filter(filtered_data[::-1, 3], size=5, mode='nearest')
    # filtered_data[-1:, 3] = 112
    # filtered_data[0, :] = [563, 624, 873, 720, 16]
    polygon_left = (np.column_stack((filtered_data[:, 0], filtered_data[:, 3])))
    polygon_right = (np.column_stack((filtered_data[::-1, 2], filtered_data[::-1, 3])))
    polygon = (np.row_stack((polygon_left, polygon_right)), np.dtype('int32'))[0]

    for i, box in enumerate(polygon):
        cv2.line(output_image, (int(polygon[i][0]), int(polygon[i][1])),
                 (int(polygon[i - 1][0]), int(polygon[i - 1][1])), (0, 255, 0), thickness=3)

    # cv2.fillPoly(output_image, polygon, (255, 0, 255))
    output = np.zeros_like(output_image, dtype=np.uint8)
    output[:, :, :] = output_image
    output = cv2.fillPoly(output, [polygon], (0, 0, 255))
    cv2.addWeighted(output, 0.5, output_image, 0.5, 0, output_image)
    # output_image = output_image[:, :, 0]

    return output_data
