
class Counter(object):
    overlap = 0.3
    between_frame_counter = 0

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

   # lines_data =
    # for i, line in enumerate(Memory.y_lines):
    #
    #     y_boxes = filtered_data[np.where(filtered_data[:, 3] == Memory.y_lines[i])]
    #     # lines_data.insert(i, filtered_data[i])
    #     print("y_boxes", y_boxes)
    #     overlap = []
    #     if len(y_boxes) > 1:
    #         for j, box in enumerate(y_boxes):
    #             ################
    #             # тут продолжить
    #             #################
    #
    #             overlap.append(area(y_boxes[j], lines_data[i-1]))
    #         lines_data[i] = y_boxes[np.where(overlap == max(overlap))][0]
    #     elif len(y_boxes) == 0:
    #         print("i", i)
    #         lines_data[i] = [lines_data[i, 0], Memory.init_box_detected[i, 1], lines_data[i, 2], Memory.y_lines[i], lines_data[i, 4]]
    #     # lines_data[i] = filtered_data[np.where(filtered_data[i, 3] == line)]
    #     else:
    #         lines_data[i] = y_boxes
    #
    #     class_rails = lines_data[i][4]
    #
    #     if class_rails == box_class.index('double central rails'):
    #         cx_right = lines_data[i][2] - Memory.previous_data[i, 2] + Memory.previous_data[i, 0]
    #         cx_left = lines_data[i][0] + Memory.previous_data[i, 2] - Memory.previous_data[i, 0]
    #
    #         # if cx1
    #         # y_double_central_rails = line[1]
    #         # x1_double_central_rails = (b[:, 0] * a[:, 1] - y_double_central_rails[:] * (b[:, 0] - a[:, 0]) - a[:, 0] * b[:, 1]) / (a[:, 1] - b[:, 1])
    #         # x2_double_central_rails = (b[:, 2] * a[:, 3] - y_double_central_rails[:] * (b[:, 2] - a[:, 2]) - a[:, 2] * b[:, 3]) / (a[:, 3] - b[:, 3])
    #         if Memory.switch_left_front >= 20:
    #             filtered_data[i, 0] = int(cx_left)
    #             # double_central_rails[:, 0] = cx_left[:]
    #             print("LEFT")
    #             # filtered_data = double_central_rails[np.where(~(cx_right <= x2_double_central_rails) & ~(cx_right >= x1_double_central_rails))]
    #         if Memory.switch_right_front >= 20:
    #             filtered_data[i, 0] = int(cx_right)
    #             # cv2.circle(output_image,
    #             #            (int(filtered_data[double_central_rails_filter][:][0][0]), y_double_central_rails), 40,
    #             #            (0, 255, 0), thickness=-1)
    #             print("RIGHT")
    #             # filtered_data[double_central_rails[np.where(~(cx_right <= x2_double_central_rails) & ~(cx_right >= x1_double_central_rails), 1, 0)]] = [x1_double_central_rails, ...]
    # Memory.current_rails_detected = True
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

        # print("left", Memory.switch_left_front)
    # if (Memory.current_rails_detected == False) & (Memory.init_box_detected == []):
    #     Memory.init_box_detected = filtered_data
    #     Memory.current_rails_detected == True

    # double_central_rails = filtered_data[filtered_data[:, 4] == 0]
    # for i, box in enumerate(double_central_rails):
    #     double_central_rails[i] = data_center_x(double_central_rails)
    # if Memory.x1_perspective != []:
    #     double_central_rails[:, 3] = Memory.x2_perspective
    #     double_central_rails_left = [double_central_rails[:, 0], double_central_rails[:, 1], (double_central_rails[:, 0] + (Memory.x2_perspective[:] - Memory.x1_perspective[:]))]
    #     double_central_rails_left = [double_central_rails[:, 0], double_central_rails[:, 1], (double_central_rails[:, 0] + (Memory.x2_perspective[:, 0] - Memory.x1_perspective[:, 0]))]
    #     filtered_data = filtered_data[np.where((double_central_rails_cx <= x2) & (double_central_rails_cx >= x1))]
    # if Memory.switch_right_front >= 20:


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

# @staticmethod
# def class_filter(norm_boxes, class_rails):
#     """
#
#     :param norm_boxes:
#     :param class_rails:
#     :return:
#     """
#     inframe_counter = 0
#
#     df = pd.DataFrame(norm_boxes, columns=['x0', 'y0', 'x1', 'y1', 'class'])
#     filter_rails = df['class'] == 3
#     central_rails = df.loc[filter_rails]
#     y0_count = central_rails['y0'].value_counts(sort=False)
#     unique_y0_filter = y0_count.values == 1
#     unique_y0_count = y0_count.loc[unique_y0_filter]
#
#     filter_unique_rails = central_rails[central_rails['y0'].isin(unique_y0_count.index.to_list())]
#
#     return filter_unique_rails
#
# def box_center_x(self, box):
#     cx = int(box[0]) + int((box[2] - box[0]) / 2)
#     return cx
#
# def box_center_y(self, box):
#     cy = box[1] + int((box[3] - box[1]) / 2)
#     return cy
#
# #####################################
# # INIT ##############################
# #####################################
# current_rails = []
#
# def init_current_rails(self, norm_boxes, class_rails, output_image):
#     inframe_counter = 0
#     # df = pd.DataFrame(norm_boxes, columns=['x0', 'y0', 'x1', 'y1', 'class'])
#     # class_rails = 'central rails'
#     filter_unique_central_rails = self.class_filter(norm_boxes, class_rails)
#     current_box = []
#     for i, box in enumerate(filter_unique_central_rails.values):
#         overlap = area(filter_unique_central_rails.values[i], filter_unique_central_rails.values[i - 1])
#
#         cx = int(self.box_center_x(box))
#         cy = int(self.box_center_y(box))
#         if inframe_counter < 5:
#             if overlap >= Counter.overlap:
#                 # uncomment
#                 cv2.circle(output_image, (cx, cy), 40, (0, 255, 255), thickness=-1)
#                 inframe_counter += 1
#                 current_box = box
#                 Memory.current_frame_rails.append(box)
#
#             else:
#                 # uncomment
#                 cv2.circle(output_image, (cx, cy), 40, (0, 100, 255), thickness=-1)
#                 inframe_counter = 0
#
#                 # break
#         # elif memory.current_rails_detected == False:
#         elif cx in range(int(current_box[0]), int(current_box[2])):
#             # инициализация
#             if overlap >= Counter.overlap:
#                 # uncomment
#                 cv2.circle(output_image, (cx, cy), 40, (0, 255, 0), thickness=-1)
#                 Memory.current_frame_rails.append(box)
#                 inframe_counter += 1
#                 current_box = box
#                 Memory.init_box_detected = box
#                 Memory.previous_box = box
#                 Memory.current_rails_detected = True
#                 return 1
#             else:
#                 # uncomment
#                 cv2.circle(output_image, (cx, cy), 40, (0, 255, 255), thickness=-1)
#                 Memory.current_frame_rails.append(box)
#
#         else:
#             # uncomment
#             cv2.circle(output_image, (cx, cy), 40, (0, 0, 255), thickness=-1)
#             return 0
#
# ###########################################################################
# # continue the current rails ##############################################
# ###########################################################################
# def continue_current_rails(self, norm_boxes, output_image):
#
#     inframe_counter = 0
#     current_box = []
#     current_rails = []
#     for i, box in enumerate(norm_boxes):
#
#         overlap = area(norm_boxes[i], norm_boxes[i - 1])
#         # overlap = area(norm_boxes[i], Memory.previous_box)
#
#         cx = self.box_center_x(box)
#         cy = self.box_center_y(box)
#         # uncomment
#         cv2.circle(output_image, (cx, cy), 20, (255, 0, 155), thickness=-1)
#
#         if cx in range(int(Memory.init_box_detected[0]), int(Memory.init_box_detected[2])):
#             Memory.previous_box = box
#             if overlap >= Counter.overlap:
#
#                 # Memory.previous_box = box
#                 # uncomment
#                 cv2.circle(output_image, (cx, cy), 40, (0, 255, 0), thickness=-1)
#                 Memory.current_frame_rails.append(box)
#
#                 # if box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
#                 #     Memory.previous_switch_right_front += 1
#                 #     Memory.between_frame_switch_front += 1
#                 #     Memory.init_box_detected = box
#                 #     Counter.between_frame_counter += 1
#                 #
#                 # elif box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
#                 #     Memory.previous_switch_right_front += 1
#                 #     Memory.between_frame_switch_front += 1
#                 #     Memory.init_box_detected = box
#                 #     Counter.between_frame_counter += 1
#                 #
#                 # elif box[4] == 'right crossing' and Memory.previous_switch_right_front >= 10:
#                 #     Memory.init_box_detected = box
#                 #
#                 # elif box[4] == 'switch left front':
#                 #     Memory.previous_switch_left_front += 1
#                 #
#                 # elif box[4] == 'left crossing' and Memory.previous_switch_left_front >= 10:
#                 #     Memory.init_box_detected = box
#                 #
#                 # elif box[4] == 'central rails':
#                 #     if self.init_current_rails(norm_boxes, box[4], output_image) == 1:
#                 #         Memory.previous_switch_left_front = 0
#                 #         Memory.previous_switch_right_front = 0
#                 #
#                 # elif box[4] == 'switch' and self.init_current_rails(norm_boxes, box[4], output_image) == 1:
#                 #     # Memory.previous_switch_left_front = 0
#                 #     # Memory.previous_switch_right_front = 0
#                 #     pass
#                 #
#                 # elif box[4] == 'double central rails':
#                 #     cx1 = self.box_center_x(box) + int((box[2] - box[0]) / 4)
#                 #     cx2 = self.box_center_x(box) - int((box[2] - box[0]) / 4)
#                 #     cy = self.box_center_y(box)
#                 #
#                 #     if cx1 in range(int(Memory.init_box_detected[0]), int(Memory.init_box_detected[2])):
#                 #         cv2.circle(output_image, (cx1, cy), 40, (0, 255, 0), thickness=-1)
#                 #         Memory.current_frame_rails.append(box)
#                 #         Memory.previous_box[0] = self.box_center_x(box)
#                 #         Memory.init_box_detected = box
#                 #         Memory.init_box_detected[0] = self.box_center_x(box)
#                 #
#                 #     else:
#                 #         cv2.circle(output_image, (cx2, cy), 40, (0, 255, 0), thickness=-1)
#                 #         Memory.current_frame_rails.append(box)
#                 #         Memory.init_box_detected = box
#                 #         Memory.previous_box[2] = self.box_center_x(box)
#
#                 #############################
#                 # old version
#                 ###########################
#                 # if box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
#                 #     Memory.previous_switch_right_front += 1
#                 #     Memory.between_frame_switch_front += 1
#                 #     Memory.init_box_detected = box
#                 #     Counter.between_frame_counter += 1
#                 #
#                 # elif box[4] == 'switch right front' and Memory.between_frame_switch_front == 0:
#                 #     Memory.previous_switch_right_front += 1
#                 #     Memory.between_frame_switch_front += 1
#                 #     Memory.init_box_detected = box
#                 #     Counter.between_frame_counter += 1
#                 #
#                 # elif box[4] == 'right crossing' and Memory.previous_switch_right_front >= 10:
#                 #     Memory.init_box_detected = box
#                 #
#                 # elif box[4] == 'switch left front':
#                 #     Memory.previous_switch_left_front += 1
#                 #
#                 # elif box[4] == 'left crossing' and Memory.previous_switch_left_front >= 10:
#                 #     Memory.init_box_detected = box
#                 #
#                 # elif box[4] == 'central rails':
#                 #     if self.init_current_rails(norm_boxes, box[4], output_image) == 1:
#                 #         Memory.previous_switch_left_front = 0
#                 #         Memory.previous_switch_right_front = 0
#                 #
#                 # elif box[4] == 'switch' and self.init_current_rails(norm_boxes, box[4], output_image) == 1:
#                 #     # Memory.previous_switch_left_front = 0
#                 #     # Memory.previous_switch_right_front = 0
#                 #     pass
#                 #
#                 # elif box[4] == 'double central rails':
#                 #     if Memory.previous_switch >= 10:
#                 #         continue
#                 #     else:
#                 #         Memory.previous_switch_left_front = 0
#                 #         Memory.previous_switch_right_front = 0
#
#             else:
#                 # uncomment
#                 cv2.circle(output_image, (cx, cy), 40, (0, 255, 200), thickness=-1)
#                 Memory.current_frame_rails.append(box)
#                 # Memory.previous_box = box
#
#         else:
#             pass
#             # uncomment
#             cv2.circle(output_image, (cx, cy), 40, (0, 0, 255), thickness=-1)
#
#

# normboxes = self.init_current_rails(normboxes, output_image)
# self.init_current_rails(normboxes, output_image)
# if not Memory.current_rails_detected:
#     self.init_current_rails(norm_boxes, "central rails", output_image)
# else:
#     self.continue_current_rails(norm_boxes, output_image)

# normboxes = Memory.current_frame_rails
# print(time.time() - timer1)
