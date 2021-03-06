import cv2
import numpy as np
import time
from tracker_utils import distinct_colors
from process_classes.utils import display_box, calculate_centr_box
from infer_utils.display_utils import *

import torch
import random
import pandas as pd
from iou import *

from counter import *
from memory import *

class display_train(object):
    def __init__(self, view_cam=False, video_loader=None, delay=0):
        # delay - Управление режимом вывода. 1 видеорежим, 0 покадровый режим
        self.delay = delay
        self.view_cam = view_cam
        self.video_loader = video_loader
        if video_loader is None:
            self.width_origin = 0
            self.height_origin = 0
            self.view_cam = False
        else:
            self.height_origin, self.width_origin = video_loader.get_origin_resolution()

        if self.width_origin == 0 or self.height_origin == 0: self.view_cam = False

        self.init_constant()

    def init_constant(self):
        self.title = "--- Glosav analytics ---"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.thickness = 2
        self.calc_lines = []

        self.org3 = (10, int(self.height_origin - 7))
        self.color3 = (0, 255, 255)

        if not self.view_cam: return
        self.init_window()

    def init_window(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1280, 720)

    def set_view_mode(self, state):
        self.view_cam = state
        self.init_constant()

    def display_train(self, calc_bilder, opt_alg):
        # color = (255, 0, 0)
        tracks = opt_alg["tracks"]
        for trk in tracks:
            for box in trk.history_box:
                display_box(self.im, box, trk.color_box)

            centr = []  # calculate_centr_box(trk.history_box, np.array([0.5, 0.5]))
            for box in trk.history_box:
                centr.append(
                    (box[2:4] + box[:2]) / 2
                )
            for i in range(1, len(centr)):
                cv2.line(self.im, tuple(np.int32(centr[i - 1])),
                         tuple(np.int32(centr[i])), trk.color_box, thickness=3)

    def display_line(self, line, boxes, class_names, height_line=64):
        # boxes = boxes[::-1]
        line = cv2.resize(line, dsize=(line.shape[1], height_line), interpolation=cv2.INTER_CUBIC)
        for box in boxes:
            box_class = int(box[5])
            temp = box[0:4].to(dtype=torch.int32).tolist()
            display_box(line, [temp[0], 0, temp[2], height_line - 1], color=self.class_colours[box_class], thickness=30)
            cv2.putText(line, class_names[box_class], (temp[0], 10),
                        self.font, 0.5, self.class_colours[box_class], thickness=2)
        return line

    # def show_frame(self, boxes, class_names, window_name ='display'):

    # top_list, bottom_list = self.video_loader.get_top_bottom_list()
    # top_list, bottom_list = top_list[::-1], bottom_list[::-1]
    # output_image = self.video_loader.get_image()
    # no_boxes=True
    # #l_top_border = self.video_loader.get_nn_top_border()

    # for i in range(0,len(boxes)):
    #     if len(boxes[i])!=0:
    #         no_boxes=False
    #     # y1 = l_top_border
    #     # y2 = l_top_border + self.video_loader.get_nn_height_line()[i]
    #     y1 = top_list[i]
    #     y2 = bottom_list[i]

    # for box in boxes[i]: box_class = int(box[5]) class_name = class_names[box_class] #if class_name not in ["double
    # left rails", "double right rails", "left rails", "right rails", "half left rails", "half right rails"]: temp =
    # box[0:3].to(dtype = torch.int32)

    #         display_box(output_image,[temp[0], y1,\
    #             temp[2],y2], color=self.class_colours[box_class],thickness=3)
    #         text = class_names[box_class] + ' ' + str(round(box[4].item(), 4))
    #         cv2.putText(output_image,text,(temp[0], y1 + 10),\
    #         self.font,0.5,self.class_colours[box_class],thickness=2)

    #     #l_top_border += self.video_loader.get_nn_height_line()[i]

    # cv2.imshow(window_name, output_image)
    # cv2.waitKey(0)

    # return output_image

    ##########################################
    # boxes in normal coordinates ############
    ##########################################
    def get_norm_box(self, boxes, class_names):
        boxes = boxes[::-1]

        top_list, bottom_list = self.video_loader.get_top_bottom_list()
        top_list, bottom_list = top_list[::-1], bottom_list[::-1]

        norm_boxes = []
        for i in range(0, len(boxes)):
            if len(boxes[i]) != 0:
                no_boxes = False
            y1 = top_list[i]
            y2 = bottom_list[i]
            for box in boxes[i]:
                box_class = int(box[5])
                class_name = class_names[box_class]
                res_box = [box[0].item(), y1, box[2].item(), y2, class_name]
                norm_boxes.append(res_box)

        return norm_boxes

    @staticmethod
    def class_filter(df, class_rails):
        """

        :param df:
        :param class_rails:
        :return:
        """
        inframe_counter = 0

        # print(df)
        filter_rails = df['class'] == class_rails
        central_rails = df.loc[filter_rails]
        # print("only center",central_rails)
        y0_count = central_rails['y0'].value_counts(sort=False)
        unique_y0_filter = y0_count.values == 1
        unique_y0_count = y0_count.loc[unique_y0_filter]

        filter_unique_rails = central_rails[central_rails['y0'].isin(unique_y0_count.index.to_list())]

        return filter_unique_rails

    def box_center_x(self, box):
        cx = int(box[0]) + int((box[2] - box[0]) / 2)
        return cx

    def box_center_y(self, box):
        cy = box[1] + int((box[3] - box[1]) / 2)
        return cy

    #####################################
    # INIT ##############################
    #####################################
    current_rails = []

    def init_current_rails(self, norm_boxes, class_rails, output_image):
        df = pd.DataFrame(norm_boxes, columns=['x0', 'y0', 'x1', 'y1', 'class'])
        df
        inframe_counter = 0

        filter_unique_central_rails = self.class_filter(df, class_rails)
        current_box = []
        filter_rails = df['class'] == class_rails
        for i, box in enumerate(df['y0']):
            Memory.previous_init_boxes = df['y0'] == i

            Memory.current_frame_rails.append(box)
            overlap = area(filter_unique_central_rails.values[i], filter_unique_central_rails.values[i - 1])

            cx = self.box_center_x(box)
            cy = self.box_center_y(box)

            if inframe_counter < 5:
                if overlap >= Counter.overlap:
                    # uncomment
                    cv2.circle(output_image, (cx, cy), 40, (0, 255, 255), thickness=-1)
                    inframe_counter += 1
                    current_box = box

                else:
                    # uncomment
                    cv2.circle(output_image, (cx, cy), 40, (0, 100, 255), thickness=-1)
                    inframe_counter = 0

                    # break
            elif cx in range(int(current_box[0]), int(current_box[2])):
                Memory.current_frame_rails[i] = box
                # инициализация
                if overlap >= Counter.overlap:

                    # uncomment
                    cv2.circle(output_image, (cx, cy), 40, (0, 255, 0), thickness=-1)
                    # Memory.current_frame_rails[i] = box
                    inframe_counter += 1
                    current_box = box
                    Memory.init_box_detected = box
                    Memory.previous_box = box
                    Memory.current_rails_detected = True
                    return 1
                else:
                    # uncomment
                    cv2.circle(output_image, (cx, cy), 40, (0, 255, 255), thickness=-1)
                    # Memory.current_frame_rails[i] = box

            else:
                # uncomment
                cv2.circle(output_image, (cx, cy), 40, (0, 0, 255), thickness=-1)
                return 0


    # def init_continue_rails(self, box, overlap, inframe_counter, output_image):
    #
    #     cx = self.box_center_x(box)
    #     cy = self.box_center_y(box)
    #     if inframe_counter < 5:
    #         # cv2.circle(output_image, (cx, cy), 40, (0, 255, 255), thickness=-1)
    #         inframe_counter += 1
    #         current_box = box
    #         Memory.current_frame_rails.append(box)
    #
    #     else:
    #         cv2.circle(output_image, (cx, cy), 40, (0, 255, 0), thickness=-1)
    #         Memory.current_frame_rails.append(box)
    #         inframe_counter += 1
    #         current_box = box
    #         Memory.init_box_detected = box
    #
    #         Memory.previous_box = box
    #         Memory.current_rails_detected = True
    #         return 1

    ###########################################################################
    # continue the current rails ##############################################
    ###########################################################################
    def continue_current_rails(self, norm_boxes, output_image):

        inframe_counter = 0
        current_box = []
        current_rails = []
        Memory.current_frame_rails.append(Memory.init_box_detected)
        for i, box in enumerate(norm_boxes):
            # Memory.current_frame_rails.append(Memory.previous_box)
            Memory.current_frame_rails.append(Memory.init_box_detected)

            # overlap = area(norm_boxes[i], norm_boxes[i - 1])
            # overlap = area(norm_boxes[i], Memory.previous_box)
            overlap = area(norm_boxes[i], norm_boxes[i-1])

            cx = self.box_center_x(box)
            cy = self.box_center_y(box)
            # uncomment
            cv2.circle(output_image, (cx, cy), 20, (255, 0, 155), thickness=-1)
            # cv2.circle(output_image, (int(Memory.previous_init_boxes[i][0]), cy), 10, (255, 0, 155), thickness=-1)
            cv2.circle(output_image, (int(Memory.current_frame_rails[i][0]), cy), 10, (255, 0, 155), thickness=-1)
            # cv2.circle(output_image, (int(Memory.previous_init_boxes[i][2]), cy), 10, (255, 0, 155), thickness=-1)
            cv2.circle(output_image, (int(Memory.current_frame_rails[i][2]), cy), 10, (255, 0, 155), thickness=-1)

            # if cx in range(int(Memory.previous_init_boxes[i][0]), int(Memory.previous_init_boxes[i][2])):
            if cx in range(int(Memory.current_frame_rails[i][0]), int(Memory.current_frame_rails[i][2])):

                # Memory.init_box_detected = boxes

                if overlap >= Counter.overlap:
                    Memory.current_frame_rails[i] = box
                    Memory.previous_box = box
                    if inframe_counter < 5:
                        # cv2.circle(output_image, (cx, cy), 40, (0, 255, 255), thickness=-1)
                        inframe_counter += 1
                        current_box = box
                        # Memory.previous_init_boxes = Memory.current_frame_rails

                        Memory.current_frame_rails = norm_boxes[i-1]
                        print("init_continue_rails", inframe_counter)

                    else:
                        cv2.circle(output_image, (cx, cy), 40, (0, 255, 0), thickness=-1)
                        Memory.current_frame_rails[i] = box
                        inframe_counter = 0
                        current_box = box
                        Memory.init_box_detected = box
                        print("sure")
                        Memory.previous_box = box

                    # Memory.previous_box = box
                    # uncomment
                    cv2.circle(output_image, (cx, cy), 40, (0, 255, 0), thickness=-1)
                    # print(box)

                    # if box[4] == 'double central rails':
                    #     x1 = int(box[2] - int(Memory.previous_box[2] - Memory.previous_box[0]))
                    #     x2 = box[2]
                    #
                    #     cy = self.box_center_y(box)
                    #     double_box2 = [x1, box[1], x2, box[3], box[4]]
                    #     double_box1 = [box[0], box[1], x1, box[3], box[4]]
                    #     cx1 = self.box_center_x(double_box1)
                    #     cx2 = self.box_center_x(double_box2)
                    #     if area(double_box2, norm_boxes[i-1]) > area(double_box1, norm_boxes[i-1]):
                    #         cv2.circle(output_image, (cx2, cy), 40, (0, 255, 0), thickness=-1)
                    #         Memory.current_frame_rails[i] = double_box2
                    #         Memory.previous_box = box
                    #         # Memory.previous_box[0] = int(box[2] - int(Memory.previous_box[2] - Memory.previous_box[0]))
                    #         # Memory.init_box_detected[0] = int(box[2] - int(Memory.previous_box[2] - Memory.previous_box[0]))
                    #     else:
                    #         # Memory.init_box_detected = boxes
                    #         Memory.previous_box = box
                    #         Memory.current_frame_rails[i] = double_box1
                    #         cv2.circle(output_image, (cx1, cy), 40, (0, 255, 0), thickness=-1)

                    # if box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
                    #     Memory.previous_switch_right_front += 1
                    #     Memory.between_frame_switch_front += 1
                    #     Memory.init_box_detected = box
                    #     Counter.between_frame_counter += 1
                    #
                    # elif box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
                    #     Memory.previous_switch_right_front += 1
                    #     Memory.between_frame_switch_front += 1
                    #     Memory.init_box_detected = box
                    #     Counter.between_frame_counter += 1
                    #     # print("ПОВОРОТ НАПРАВО", Counter.between_frame_counter)
                    #
                    # elif box[4] == 'right crossing' and Memory.previous_switch_right_front >= 10:
                    #     Memory.init_box_detected = box
                    #     # print("ПОВОРОТ НАПРАВО", Memory.previous_switch_right_front)
                    #
                    # elif box[4] == 'switch left front':
                    #     Memory.previous_switch_left_front += 1
                    #
                    # elif box[4] == 'left crossing' and Memory.previous_switch_left_front >= 10:
                    #     Memory.init_box_detected = box
                    #
                    # elif box[4] == 'central rails':
                    #     if self.init_current_rails(norm_boxes, box[4], output_image) == 1:
                    #         # print("INIT OUT")
                    #         Memory.previous_switch_left_front = 0
                    #         Memory.previous_switch_right_front = 0
                    #
                    # elif box[4] == 'switch' and self.init_current_rails(norm_boxes, box[4], output_image) == 1:
                    #     # Memory.previous_switch_left_front = 0
                    #     # Memory.previous_switch_right_front = 0
                    #     pass
                    #
                    ######################3
                    ##########3
                    ##################################
                    #################
                    ###############################################################
                        #
                        # else:
                        #     cv2.circle(output_image, (x2, cy), 40, (0, 255, 0), thickness=-1)
                        #     Memory.current_frame_rails.append(box)
                        #     Memory.init_box_detected = box
                        #     Memory.previous_box[2] = self.box_center_x(box)

                    #############################
                    # old version
                    ###########################
                    # if box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
                    #     Memory.previous_switch_right_front += 1
                    #     Memory.between_frame_switch_front += 1
                    #     Memory.init_box_detected = box
                    #     Counter.between_frame_counter += 1
                    #
                    #     # print("ПОВОРОТ НАПРАВО", Counter.between_frame_counter)
                    #     print("ПОВОРОТ НАПРАВО", Memory.previous_switch_right_front)
                    #
                    # elif box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
                    #     Memory.previous_switch_right_front += 1
                    #     Memory.between_frame_switch_front += 1
                    #     Memory.init_box_detected = box
                    #     Counter.between_frame_counter += 1
                    #     # print("ПОВОРОТ НАПРАВО", Counter.between_frame_counter)
                    #
                    # elif box[4] == 'right crossing' and Memory.previous_switch_right_front >= 10:
                    #     Memory.init_box_detected = box
                    #     # print("ПОВОРОТ НАПРАВО", Memory.previous_switch_right_front)
                    #
                    # elif box[4] == 'switch left front':
                    #     Memory.previous_switch_left_front += 1
                    #
                    # elif box[4] == 'left crossing' and Memory.previous_switch_left_front >= 10:
                    #     Memory.init_box_detected = box
                    #
                    # elif box[4] == 'central rails':
                    #     if self.init_current_rails(norm_boxes, box[4], output_image) == 1:
                    #         # print("INIT OUT")
                    #         Memory.previous_switch_left_front = 0
                    #         Memory.previous_switch_right_front = 0
                    #
                    # elif box[4] == 'switch' and self.init_current_rails(norm_boxes, box[4], output_image) == 1:
                    #     # Memory.previous_switch_left_front = 0
                    #     # Memory.previous_switch_right_front = 0
                    #     pass
                    #
                    # elif box[4] == 'double central rails':
                    #     if Memory.previous_switch >= 10:
                    #         continue
                    #     else:
                    #         Memory.previous_switch_left_front = 0
                    #         Memory.previous_switch_right_front = 0

                else:
                    # uncomment
                    cv2.circle(output_image, (cx, cy), 40, (0, 255, 200), thickness=-1)
                    Memory.current_frame_rails[i] = box
                    # Memory.previous_box = box

            else:
                cv2.circle(output_image, (cx, cy), 40, (0, 0, 255), thickness=-1)

    ###########################################################################
    # current rails in absolute coordinates #########################
    ###########################################################################

    def show_current_frame(self, normboxes, class_names, window_name='current'):

        # Memory.current_frame_rails = []
        # normboxes = normboxes[::-1]
        top_list, bottom_list = self.video_loader.get_top_bottom_list()
        # top_list, bottom_list = top_list[::-1], bottom_list[::-1]
        output_image = self.video_loader.get_image()
        no_boxes = True
        timer1 = time.time()
        # normboxes = self.init_current_rails(normboxes, output_image)
        # self.init_current_rails(normboxes, output_image)
        if not Memory.current_rails_detected:

            self.init_current_rails(normboxes, "central rails", output_image)
        else:
            self.continue_current_rails(normboxes, output_image)

        # normboxes = Memory.current_frame_rails
        # print(time.time() - timer1)

        for box in normboxes:
            ###############
            # вывод
            ###############
            #uncomment
            # box[4] = "current rails"

            if box[4] == "current rails":
                display_box(output_image, [box[0], box[1],
                                           box[2], box[3]], color=(0, 255, 0), thickness=3)
                text = box[4]
                cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                            self.font, 0.5, (0, 255, 0), thickness=2)
            else:
                box_class = [
                    "double central rails", "double left rails", "double right rails", "central rails", "left rails",
                    "right rails", "half left rails", "half right rails", "switch right back", "switch left back",
                    "switch right front",
                    "switch left front", "switch", "left crossing", "right crossing", "double cross"
                ]

                display_box(output_image, [box[0], box[1], box[2], box[3]],
                            color=self.class_colours[box_class.index(box[4])], thickness=3)
                text = box[4]
                cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                            self.font, 0.5, self.class_colours[box_class.index(box[4])], thickness=2)

        return output_image

    # def show_filtred_frame(self, boxes, class_names, window_name ='display1'):
    # top_list, bottom_list = self.video_loader.get_top_bottom_list()
    # top_list, bottom_list = top_list[::-1], bottom_list[::-1]
    # output_image = self.video_loader.get_image()
    # no_boxes=True
    # #l_top_border = self.video_loader.get_nn_top_border()

    # #boxes_counter = 0
    # # inframe_counter = 0
    # # x_counter = 0
    # for i in range(0,len(boxes)):
    #     if len(boxes[i])!=0:
    #         no_boxes=False
    #     # y1 = l_top_border
    #     # y2 = l_top_border + self.video_loader.get_nn_height_line()[i]
    #     y1 = top_list[i]
    #     y2 = bottom_list[i]

    #     for box in boxes[i]:
    #         box_class = int(box[5])
    #         class_name = class_names[box_class]
    #         #if class_name not in ["double left rails", "double right rails", "left rails", "right rails", "half left rails", "half right rails"]:
    #         temp = box[0:3].to(dtype = torch.int32)

    #         #if toplist[i + 1] == bottom_list[i] & abs(boxes[i][0] - boxes[i + 1][0]) & abs(boxes[i][3] - boxes[i + 1][4])
    #         cx = temp[0].item() + int((temp[2].item() - temp[0].item())/2)
    #         cy = y1 + int((y2-y1)/2)
    #         #print(class_name)

    #         #if class_name == "central rails" & inframe_counter >= 5:
    #         #if class_name == "central rails":

    #             # inframe_counter += 1
    #             #counter.between_frame_counter += 1
    #             # print("inframe", inframe_counter)
    #             # print("between_frame", counter.between_frame_counter)

    #             #between_frame_counter += 1

    #         #if inframe_counter >= 5:
    #         #    class_name == "current rails"
    #         #    cv2.circle(output_image,(cx, cy), 40, (0,255,0), thickness=-1)
    #         #if class_name == "switch right front" | class_name == "switch left front" :
    #             #between_frame_counter += 1

    #         display_box(output_image,[temp[0], y1,\
    #             temp[2],y2], color=self.class_colours[box_class],thickness=3)
    #         text = class_names[box_class] + ' ' + str(round(box[4].item(), 4))
    #         cv2.putText(output_image,text,(temp[0], y1 + 10),\
    #         self.font,0.5,self.class_colours[box_class],thickness=2)

    # cv2.imshow(window_name, output_image)
    # cv2.waitKey(0)

    # return output_image
    def make_norm_data(self, norm_boxes, frame_name, class_names, window_name='display'):
        output_image = self.video_loader.get_image()
        no_boxes = True

        df = pd.DataFrame(norm_boxes, columns=['x0', 'y0', 'x1', 'y1', 'class', 'num_class'])
        # pred_lists = []
        indexes = 0
        for box in norm_boxes:

            # pred_list = [class_names[box_class], cx, cy, indexes]
            # pred_lists.append(pred_list)
            pred_list = box
            df.loc[indexes] = pred_list

            if box[4] == "current rails":
                display_box(output_image, [box[0], box[1],
                                           box[2], box[3]], color=(0, 255, 0), thickness=3)
                text = box[4]
                cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                            self.font, 0.5, (0, 255, 0), thickness=2)
            else:
                box_class = [
                    "double central rails", "double left rails", "double right rails", "central rails",
                    "left rails",
                    "right rails", "half left rails", "half right rails", "switch right back", "switch left back",
                    "switch right front",
                    "switch left front", "switch", "left crossing", "right crossing", "double cross"
                ]

                display_box(output_image, [box[0], box[1], box[2], box[3]],
                            color=self.class_colours[box_class.index(box[4])], thickness=3)
                text = box[4]
                cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                            self.font, 0.5, self.class_colours[box_class.index(box[4])], thickness=2)

        cv2.imwrite('video/' + frame_name + '.png', output_image)
        df.to_csv('video2/' + frame_name + '.csv', index_label='num')
        # cv2.imshow(window_name, output_image)
        # cv2.waitKey(0)

    def make_data(self, boxes, frame_name, class_names, window_name='display'):
        top_list, bottom_list = self.video_loader.get_top_bottom_list()
        output_image = self.video_loader.get_image()
        no_boxes = True

        df = pd.DataFrame(columns=['label', 'cx', 'cy'])
        # pred_lists = []
        indexes = 0
        for i in range(0, len(boxes)):
            if len(boxes[i]) != 0:
                no_boxes = False
            y1 = top_list[i]
            y2 = bottom_list[i]

            for box in boxes[i]:
                box_class = int(box[5])
                temp = box[0:3].to(dtype=torch.int32)
                cx = temp[0].item() + int((temp[2].item() - temp[0].item()) / 2)
                cy = y1 + int((y2 - y1) / 2)

                # pred_list = [class_names[box_class], cx, cy, indexes]
                # pred_lists.append(pred_list)
                pred_list = [class_names[box_class], cx, cy]
                df.loc[indexes] = pred_list

                display_box(output_image, [temp[0], y1,
                                           temp[2], y2], color=self.class_colours[box_class], thickness=3)

                text = class_names[box_class]
                cv2.putText(output_image, text, (temp[0], y1 + 10),
                            self.font, 0.5, self.class_colours[box_class], thickness=2)
                cv2.putText(output_image, str(indexes), (temp[2], y1 + 10),
                            self.font, 0.5, (0, 0, 0), thickness=1)

                indexes += 1

        cv2.imwrite('/media/user3070/data/teplovoz_table_data/video2/' + frame_name + '.png', output_image)
        df.to_csv('/media/user3070/data/teplovoz_table_data/video2/' + frame_name + '.csv', index_label='num')
        cv2.imshow(window_name, output_image)
        cv2.waitKey(0)

    def set_class_colors(self, num_classes):
        self.class_colours = []
        for i in range(20):
            self.class_colours.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def set_str_class_colors(self, num_classes):
        self.class_colours = []
        for i in range(20):
            self.class_colours.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def display(self, calc_bilder, opt_alg):  # frame_counter, track, del_track):
        if not self.view_cam: return True

        self.set_image()

        # self.display_train(calc_bilder, opt_alg)
        self.display_box(calc_bilder, opt_alg)

        im = cv2.putText(self.im, 'Frame: %d' % opt_alg["frames"], self.org3, self.font, self.fontScale, self.color3,
                         self.thickness, cv2.LINE_AA)

        for calc_object in calc_bilder.objects:
            displayed = calc_object.get_displayed({})
            lines = displayed.get("sides", [])
            for line in lines:
                cv2.line(self.im, line[0], line[1], line[2], thickness=2)

        if cv2.getWindowProperty(self.title, cv2.WND_PROP_AUTOSIZE) != 0:
            self.init_window()
        cv2.imshow(self.title, im)

        key = cv2.waitKey(self.delay)
        if key & 0xFF == 13 or key & 0xFF == 141:  # Если нажали Enter то перейти в видеорежим
            self.delay = 20
        if key & 0xFF == 32:  # Если нажали Space то перейти в покадровый режим
            self.delay = 0
        if key & 0xFF == ord('q') or key & 0xFF == 202:  # Если нажали Q то выйти
            return False
        return True

    def set_image(self):
        if self.video_loader is None or not self.view_cam:
            self.im = None
        else:
            # self.im  = loader.get_image()
            self.im = self.video_loader.get_image_origin()
