import cv2
import pprint
import numpy as np
import time
from tracker_utils import distinct_colors
from process_classes.utils import display_box, calculate_centr_box
from infer_utils.display_utils import *
import json
import torch
import random
import pandas as pd
from iou import *

from counter import *
from memory import *
from main import *

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

    ###########################################################################
    # current rails in absolute coordinates #########################
    ###########################################################################

    def show_current_frame(self, norm_boxes, class_names, window_name='current'):

        output_image = self.video_loader.get_image()
        frame_width = InputPlug.FrameWidth
        frame_height = InputPlug.FrameHeight
        rail_boxes = norm_boxes
        crop_coors = np.int16(InputPlug.CropCoors * frame_height)
        timer1 = time.time()
        filtred_boxes = analytics(rail_boxes, crop_coors, frame_width, frame_height, output_image)

        # polygon = analytics(rail_boxes, crop_coors, frame_width, frame_height, output_image)
        # print(time.time() - timer1)

        no_boxes = True

        # uncomment
        # вывод боксов
        for box in norm_boxes:
            box_class = int(box[4])
            class_name = class_names[box_class]
            ###############
            # вывод
            ###############
            display_box(output_image, [box[0], box[1], box[2], box[3]],
                        color=self.class_colours[box_class], thickness=1)
            text = class_names[box_class]
            cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                        self.font, 0.5, self.class_colours[box_class], thickness=1)
        #
        for box in filtred_boxes:

            display_box(output_image, [box[0], box[1],
                                       box[2], box[3]], color=(0, 255, 0), thickness=2)

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

    #         #if class_name == "central rails" & inframe_counter >= 5:
    #         #if class_name == "central rails":

    #             # inframe_counter += 1
    #             #counter.between_frame_counter += 1


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
