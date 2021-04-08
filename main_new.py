from pprint import pprint
import numpy as np
import cv2
from memory import *
from input_plug import *
import scipy.ndimage.filters as filt
import bottleneck as bn
from scipy.ndimage.filters import gaussian_filter

box_class = ["double central rails", "double left rails", "double right rails", "central rails", "left rails",
             "right rails", "half left rails", "half right rails", "switch right back", "switch left back",
             "switch right front",
             "switch left front", "switch", "left crossing", "right crossing", "double cross", "reserve 1", "reserve 2"]


def box_center_x(box):
    cx = int(box[0]) + int((box[2] - box[0]) / 2)
    return cx


def box_center_y(box):
    cy = box[1] + int((box[3] - box[1]) / 2)
    return cy


def data_center_x(filtered_data):
    cx = filtered_data[:, 0] + (filtered_data[:, 2] - filtered_data[:, 0]) / 2
    return cx


# x = (x2y1 - y(x2 - x1) - x1y2) / (y1 - y2)
def equation_line(x1, y1, x2, y2, y):
    x = (x2 * y1 - y * (x2 - x1) - x1 * y2) / (y1 - y2)
    # x = (x2[:] * y1[:] - y[:] * (x2[:] - x1[:]) - x1[:] * y2[:]) / (y1[:] - y2[:])
    return x


def perspective(filtered_data, class_rails):
    central_rails = np.copy(filtered_data[np.where(filtered_data[:, 4] == 3)])
    y = filtered_data[:, 3]

    if len(central_rails) >= 10 & \
            ((not any(class_rails == box_class.index("switch right front"))) |
             (not any(class_rails == box_class.index("switch left front")))):
        Memory.a_perspective = central_rails[[0]]
        Memory.b_perspective = central_rails[[-1]]

        Memory.x1_perspective = equation_line(Memory.a_perspective[:, 0], Memory.a_perspective[:, 3],
                                              Memory.b_perspective[:, 0], Memory.b_perspective[:, 3], y)
        Memory.x2_perspective = equation_line(Memory.a_perspective[:, 2], Memory.a_perspective[:, 3],
                                              Memory.b_perspective[:, 2], Memory.b_perspective[:, 3], y)
        cx = data_center_x(filtered_data)
        filtered_data = filtered_data[(cx <= Memory.x2_perspective) & (cx >= Memory.x1_perspective)]
        Memory.switch_right_front = 0
        Memory.switch_left_front = 0
        # Memory.right_crossing = 0
        # Memory.left_crossing = 0
        if (not any(class_rails == box_class.index("switch right front"))) | \
                (not any(class_rails == box_class.index("switch left front"))):
            Memory.right_crossing = 0
            Memory.left_crossing = 0
        # print("<СБРОС>")
    return filtered_data

##########################################
# boxes in normal coordinates ############
##########################################
def get_norm_box(boxes, class_names):
    # crop_coors = np.int16(boxes[ * InputPlug.FrameHeight)
    boxes = boxes[::-1]
    # top_list, bottom_list = self.video_loader.get_top_bottom_list()
    # top_list, bottom_list = top_list[::-1], bottom_list[::-1]
    # InputPlug.CropCoors = InputPlug.CropCoors * InputPlug.FrameHeight
    # InputPlug.CropCoors = np.int16(InputPlug.CropCoors * InputPlug.FrameHeight)
    crop_coors = np.int16(InputPlug.CropCoors * InputPlug.FrameHeight)
    top_list = crop_coors[:, 0]
    bottom_list = crop_coors[:, 1]

    norm_boxes = []
    for i in range(0, len(boxes)):
        if len(boxes[i]) != 0:
            no_boxes = False
        y1 = top_list[i]
        # y2 = top_list[i]
        y2 = bottom_list[i]
        # y1 = bottom_list[i]
        for box in boxes[i]:
            # box[0] = box[0] * InputPlug.FrameWidth
            # box[1] = box[1] * InputPlug.FrameHeight
            # box[2] = box[2] * InputPlug.FrameWidth
            # box[3] = box[3] * InputPlug.FrameHeight

            if int(box[5]) == 0:
                right_double_box = [(box[2].item() - (i * i * 0.2464 - i * 15.45 + 309.5)), y1, box[2].item(), y2, 17]
                # right_double_box = [(box[2].item() - (Memory.x2[i] - Memory.x1[i])), y1, box[2].item(), y2, 16]

                # left_double_box = [box[0].item(), y1, (box[0].item() + (Memory.x2[i] - Memory.x1[i])), y2, 17]
                left_double_box = [box[0].item(), y1, (box[0].item() + (i * i * 0.2464 - i * 15.45 + 309.5)), y2, 16]

                norm_boxes.append(right_double_box)
                norm_boxes.append(left_double_box)
            elif int(box[5]) == 12:
                # i*i*0.2464-i*15.45+309.5
                # switch_box = [(self.box_center_x(box) - (Memory.x2[i] - Memory.x1[i])/2), y1, (self.box_center_x(box) + (Memory.x2[i] - Memory.x1[i])/2), y2, 16]
                switch_box = [(box_center_x(box) - (i * i * 0.2464 - i * 15.45 + 309.5) / 2), y1,
                              (box_center_x(box) + (i * i * 0.2464 - i * 15.45 + 309.5) / 2), y2, 16]
                norm_boxes.append(switch_box)
            else:
                res_box = [box[0].item(), y1, box[2].item(), y2, int(box[5])]
                norm_boxes.append(res_box)

    return norm_boxes


def analytics(input_data, output_image):
    data = np.array(input_data, dtype=int)
    data = data.reshape(-1, 5)

    # data[:, 0] = np.int16(data[:, 0] * InputPlug.FrameWidth)
    # data[:, 2] = np.int16(data[:, 2] * InputPlug.FrameWidth)
    # data[:, 1] = np.int16(data[:, 1] * InputPlug.FrameHeight)
    # data[:, 3] = np.int16(data[:, 3] * InputPlug.FrameHeight)

    class_rails = data[:, 4]
    class_rails_filter = ((class_rails == 0) | (class_rails == 3) | ((class_rails != 15) & (class_rails >= 8)))
    filtered_data = data[class_rails_filter]
    filtered_data[0, :] = [563, 624, 873, 720, 16]

    if Memory.previous_data == []:
        Memory.previous_data = np.array(filtered_data, dtype=int)

    perspective(filtered_data, class_rails)

    class_rails = filtered_data[:, 4]

    # 0: "double central rails",
    # 1: "double left rails",
    # 2: "double right rails",
    # 3: "central rails",
    # 4: "left rails",
    # 5: "right rails",
    # 6: "half left rails",
    # 7: "half right rails",
    # 8: "switch right back",
    # 9: "switch left back",
    # 10: "switch right front",
    # 11: "switch left front",
    # 12: "switch",
    # 13: "left crossing",
    # 14: "right crossing",
    # 15: "double cross",
    # 16: "reserve 1",
    # 17: "reserve 2"

    if any(class_rails == box_class.index("switch right front")):
        Memory.switch_right_front += 1
        # print("right", Memory.switch_right_front)

    elif any(class_rails == box_class.index("switch left front")):
        Memory.switch_left_front += 1
        # print("right", Memory.switch_left_front)

    elif len(class_rails == box_class.index("right crossing")) >= 3:
        Memory.right_crossing += 1
        # print("right crossing", Memory.right_crossing)

    elif len(class_rails == box_class.index("left crossing")) >= 3:
        Memory.left_crossing += 1
        # print("right crossing", Memory.left_crossing)

    elif len(class_rails == box_class.index("reserve 1")) >= 3:
        Memory.left_crossing += 1
        # print("right crossing", Memory.left_crossing)

    elif len(class_rails == box_class.index("reserve 2")) >= 3:
        Memory.left_crossing += 1
        # print("right crossing", Memory.left_crossing)

    if (Memory.switch_right_front >= 10) | (Memory.right_crossing >= 10):
        # выкидываем reserve 1
        right_double_box = filtered_data[filtered_data[:, 4] != 16]
        # выкидываем left crossing
        right_crossing_box = right_double_box[right_double_box[:, 4] != 13]
        filtered_data = right_crossing_box
        # print("right", Memory.right_crossing, Memory.switch_right_front)

    elif (Memory.switch_left_front >= 10) | (Memory.left_crossing >= 10):
        # выкидываем reserve 2
        left_double_box = filtered_data[filtered_data[:, 4] != 17]
        # выкидываем right crossing
        left_crossing_box = left_double_box[left_double_box[:, 4] != 14]
        filtered_data = left_crossing_box
        # print("left", Memory.left_crossing, Memory.switch_left_front)

    else:
        cx = data_center_x(filtered_data)
        # cx = cx.reshape(-1, 5)
        filtered_data = filtered_data[(cx <= Memory.x2_perspective) & (cx >= Memory.x1_perspective)]
    #     filtered_data[0, :] = [563, 624, 873, 720, 16]
    #     cx = data_center_x(filtered_data)
    #     average_cx = np.int16(bn.move_mean(cx, window=2, min_count=1))
    #     average_filter = (abs(cx - average_cx) <= 15)
    #     # print(abs(cx - average_cx))
    #     filtered_data = filtered_data[average_filter]
    # perspective(filtered_data, class_rails)

    filtered_data[0, :] = [563, 624, 873, 720, 16]
    cx = data_center_x(filtered_data)
    average_cx = np.int16(bn.move_mean(cx, window=3, min_count=1))
    average_filter = (abs(cx - average_cx) <= 15)
    # print(abs(cx - average_cx))
    filtered_data = filtered_data[average_filter]
    # return filtered_data
    output_boxes = np.copy(filtered_data)

    # for i in range(len(x1)):
    #     cv2.circle(output_image, (int(x1[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    #     cv2.circle(output_image, (int(x2[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)
    #     cv2.circle(output_image, (int(cx[i]), int(y[i])), 10, (0, 255, 0), thickness=-1)

    polygon_left = (np.column_stack((filtered_data[:, 0], filtered_data[:, 3])))
    polygon_right = (np.column_stack((filtered_data[::-1, 2], filtered_data[::-1, 3])))
    polygon = (np.row_stack((polygon_left, polygon_right)), np.dtype('int32'))[0]

    # Контур полигона без сглаживания
    # for i, box in enumerate(polygon):
    #     cv2.line(output_image, (int(polygon[i][0]), int(polygon[i][1])), (int(polygon[i-1][0]), int(polygon[i-1][1])), (255, 0, 255), thickness=3)
    filtered_data[:, 0] = filt.uniform_filter(filtered_data[:, 0], size=5, mode='nearest')
    # filtered_data[:, 0] = np.int16(bn.move_mean(filtered_data[:, 0], window=3, min_count=1))
    # filtered_data[:, 0] = gaussian_filter(filtered_data[:, 0], sigma=(5), mode='nearest')
    filtered_data[:, 3] = filt.uniform_filter(filtered_data[:, 3], size=5, mode='nearest')
    # filtered_data[:, 3] = np.int16(bn.move_mean(filtered_data[:, 3], window=3, min_count=1))
    filtered_data[::-1, 2] = filt.uniform_filter(filtered_data[::-1, 2], size=5, mode='nearest')
    # filtered_data[::-1, 2] = gaussian_filter(filtered_data[::-1, 2], sigma=(5), mode='nearest')
    # filtered_data[::-1, 3] = filt.uniform_filter(filtered_data[::-1, 3], size=5, mode='nearest')
    # filtered_data[-1:, :] = filtered_data[-1:, :]
    filtered_data[-1:, 3] = filtered_data[-1:, 1]
    filtered_data[0, :] = [563, 624, 873, 720, 16]
    polygon_left = (np.column_stack((filtered_data[:, 0], filtered_data[:, 3])))
    polygon_right = (np.column_stack((filtered_data[::-1, 2], filtered_data[::-1, 3])))
    polygon = (np.row_stack((polygon_left, polygon_right)), np.dtype('int32'))[0]
    # Сглаженный контур полигона
    # for i, box in enumerate(polygon):
    #     cv2.line(output_image, (int(polygon[i][0]), int(polygon[i][1])),
    #              (int(polygon[i - 1][0]), int(polygon[i - 1][1])), (0, 255, 0), thickness=3)
    # Рисуем полигон
    # cv2.fillPoly(output_image, polygon, (255, 0, 255))
    # Полупрозрачный
    output = np.zeros_like(output_image, dtype=np.uint8)
    output[:, :, :] = output_image
    output = cv2.fillPoly(output, [polygon], (0, 0, 255))
    cv2.addWeighted(output, 0.5, output_image, 0.5, 0, output_image)
    # output_image = output_image[:, :, 0]
    polygon = np.float32(polygon)
    polygon[:, 0] = polygon[:, 0] / InputPlug.FrameWidth
    polygon[:, 1] = polygon[:, 1] / InputPlug.FrameHeight

    return output_boxes

