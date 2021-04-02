# -------------------------------------------------------------------------------------
#
# detector-counter for testing on files
#
# --------------------------------------------------------------------------------------
import json
import os
import pickle
import random
import sys
import threading
import time

from infer_utils.data_loader import video_loader
from infer_utils.display_train import display_train
# from infer_utils.display_train_old import display_train
# from infer_utils.display_train_new import display_train
# from infer_utils.display_train_kmeans import display_train
# from infer_utils.display_train_numpy import display_train
# from infer_utils.display_train_overlap import display_train
# from infer_utils.display_train_pandas import display_train
from infer_utils.video_writer import *
from main import analytics
from process_classes.Class_pipeline import Class_pipeline
from process_classes.utils import display_box
from tracker_utils import *
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression  # , non_max_suppression1
from counter import *
from memory import *

pth = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(str(pth))

# import parser_dataset
task_stat = {
    "Не начат": "NotStarted",
    "Запланирован к выполнению": "Planned",
    "Выполняется": "Running",
    "Завершен": "Finished",
    "Завершен с ошибкой": "Error"
}

mutex = threading.Lock()
hot_cake = False


class Task(threading.Thread):
    def __init__(self, opt):
        threading.Thread.__init__(self)
        self.opt = opt
        self.ret_opt = {}
        self.is_run = True
        self.hot_cake = False
        self.status_task = task_stat["Не начат"]
        self.delay = 0

        self.ret_opt["frame_counter"] = 0
        self.opt["frame_count"] = 100

    def stop(self):
        self.is_run = False
        time.sleep(0.1)

    def status(self):
        return json.dumps({
            "Status": self.status_task,
            "Progress": round(float(self.ret_opt["frame_counter"]) / self.opt["frame_count"], 2)
        })

    def run(self):
        global hot_cake, task_stat

        self.status_task = task_stat["Запланирован к выполнению"]

        while self.is_run and (not self.hot_cake):
            mutex.acquire()
            if not hot_cake:
                self.hot_cake = hot_cake = True
            mutex.release()
            time.sleep(0.1)

        if self.hot_cake:
            self.status_task = task_stat["Выполняется"]

            # try:
            ret = self.detect()
            # except Exception as e:
            #     print("Except test on video %s" % self.opt["VideoFile"])
            #
            #     ret = False

            hot_cake = False

        if not self.is_run: ret = True

        self.status_task = task_stat["Завершен"] if ret else task_stat["Завершен с ошибкой"]

    ##############################################

    def detect(self):
        print("Start test on video %s" % self.opt["VideoFile"])
        # time.sleep(5)
        # return True
        # ############## Load model ####################
        device = select_device(self.opt["gpu_id"])
        model = attempt_load(self.opt["checkpoint"], map_location=device)
        if self.opt["half"]:
            model.half()  # to FP16
        model.eval()
        ###############################################
        '''
        ############ scripted model####################
        sample = torch.rand(1, 3, 300, 300).to(device)
        scripted_model = torch.jit.trace(model, sample)
        scripted_model.save('./checkpoints/ssd300.pth')
        ###############################################
        '''

        # ############## Create Data Loader ###################
        try:
            self.opt["loader"] = video_loader(self.opt["VideoFile"], (self.opt["work_height"], self.opt["work_width"]),
                                              self.opt["batch_size"], self.opt["half"])
        except Exception as e:
            print(e)
            sys.exit(0)
        self.opt["fps"] = self.opt["loader"].get_fps()
        self.opt["frame_count"] = self.opt["loader"].get_frame_count()
        self.opt["height_origin"], self.opt["width_origin"] = self.opt["loader"].get_origin_resolution()
        ###############################################
        # Algorithm variables #########################
        # mult = torch.tensor([
        #     float(self.opt["width_origin"]) / self.opt["work_width"],
        #     float(self.opt["height_origin"]) / self.opt["work_height"],
        #     float(self.opt["width_origin"]) / self.opt["work_width"],
        #     float(self.opt["height_origin"]) / self.opt["work_height"],
        #     1, 1])

        self.ret_opt["frame_counter"] = 0
        work_time = []

        cap = cv2.VideoCapture(0)
        # out = cv2.VideoWriter('output.mp4', -1,20.0, (1280, 720))
        out = Video_holder('sim_res.avi', 1280, 720)

        detect_pipeline = Class_pipeline(self.opt, self.opt["scenarios"], None)
        #################
        video_name = opt["VideoFile"].split('/')[-1][:-4]
        # Get names
        names = model.module.names if hasattr(model, 'module') else model.names
        # with open("/home/evgeny/work/datasets/yolo_classes.json", "w") as write_file:
        #    json.dump(names, write_file)
        if self.opt["classes"] != None:
            classes = [names.index(x) for x in self.opt["classes"]]
        else:
            classes = None
        # classes = [0]#list(range(10))
        # classes = None################################################################################################3
        ############# display settings ################
        # display = frame_display_multi_line(self.opt["view_cam"], self.opt["loader"])
        display = display_train(self.opt["view_cam"], self.opt["loader"])
        display1 = display_train(self.opt["view_cam"], self.opt["loader"])

        ###############################################
        display.set_str_class_colors(len(self.opt["classes"]))
        display1.set_str_class_colors(len(self.opt["classes"]))
        ################################################################################################3
        # self.opt["loader"].set_frame(8000)

        # ###############################################################################################3

        self.ret_opt["working_time"] = time.time()
        Memory.between_frame_switch_front = 0
        while (self.opt["loader"].next_frame()):
            Memory.between_frame_switch_front = 0
            Counter.between_frame_counter = 0
            if self.is_run == False:
                break

            self.ret_opt["frame_counter"] = self.opt["loader"].get_read_frames()

            if self.opt["view_cam"]:
                if self.ret_opt["frame_counter"] < 00:
                    display.set_view_mode(False)
                    display1.set_view_mode(False)
                else:
                    display.set_view_mode(True)
                    display1.set_view_mode(True)

            if self.ret_opt["frame_counter"] % 1000 < self.opt["batch_size"]:
                print(self.ret_opt["frame_counter"])
            # Load image as tenzor ##############
            # image = self.opt["loader"].get_t()
            ###########################################
            #######################################
            t1 = time.time()

            # nn_output = []
            # for line_input in self.opt['loader'].get_nn_input():
            #     pred = model(line_input.unsqueeze(0), augment=False)[0] # Inference
            #     pred = non_max_suppression(pred, self.opt["min_score"], self.opt["overlap"], agnostic=False, classes = None) # Apply NMS
            #     pred = [x.detach().to('cpu') if x is not None else torch.tensor([]).view(-1,6)
            #             for x in pred]
            #     nn_output.extend(pred)

            nn_input = torch.zeros([1, 3, 64, 1280], dtype=torch.float16).cuda()
            for line_input in self.opt['loader'].get_nn_input():
                nn_input = torch.cat([nn_input, line_input.unsqueeze(0)])
            nn_input = nn_input[1:]
            pred = model(nn_input, augment=False)[0]
            pred = non_max_suppression(pred, self.opt["min_score"], self.opt["overlap"], agnostic=False, classes=None)
            nn_output = [x.detach().to('cpu') if x is not None else torch.tensor([]).view(-1, 6)
                         for x in pred]

            hn = len(nn_output) // self.opt["batch_size"]

            for j in range(self.opt["batch_size"]):
                boxes = nn_output[j * hn:(j + 1) * hn]  # [hn * N * 6]
                with open(f"data_{j}.pkl", 'wb') as f:
                    pickle.dump(boxes, f)
                frame = self.ret_opt["frame_counter"] - self.opt["batch_size"] + j
                frame_name = video_name + '_frame' + str(self.ret_opt["frame_counter"])

                # sort_predictions = display.sort_boxes(boxes,self.opt["classes"])
                # display.show_boxes(sort_predictions, self.opt["classes"])

                #
                # output_image = display.show_filtred_frame(boxes, self.opt["classes"]) #unblock

                norm_boxes = display.get_norm_box(boxes, self.opt["classes"])

                # display.make_norm_data(norm_boxes, frame_name, self.opt["classes"])
                # output_image = display.show_filtred_frame(boxes, self.opt["classes"])

                # prev version
                # output_image = display.show_current_frame(norm_boxes, self.opt["classes"])
                # output_image2 = self.video_loader.get_image()
                output_image = display.show_current_frame(norm_boxes, self.opt["classes"])
                # class_colours = []
                # for i in range(20):
                #     class_colours.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                # for box in output_boxes:
                #     box_class = int(box[4])
                #     if box[4] == "current rails":
                #         display_box(output_image, [box[0], box[1],
                #                                    box[2], box[3]], color=(0, 255, 0), thickness=3)
                #         text = box[4]
                #         cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
                #     else:
                #         display_box(output_image, [box[0], box[1], box[2], box[3]],
                #                     color=class_colours[box_class], thickness=3)
                #         text = self.opt["classes"][box_class]
                #         cv2.putText(output_image, text, (int(box[0]), int(box[1]) + 10),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colours[box_class], thickness=2)
                #
                # cv2.imshow('window_name', output_image)
                cv2.imshow('window_name', output_image)
                # cv2.waitKey(5)
                key = cv2.waitKey(self.delay)
                if key & 0xFF == 13 or key & 0xFF == 141:  # Если нажали Enter то перейти в видеорежим
                    self.delay = 20
                if key & 0xFF == 32:  # Если нажали Space то перейти в покадровый режим
                    self.delay = 0
                if key & 0xFF == ord('q') or key & 0xFF == 202:  # Если нажали Q то выйти
                    return False
                # return True

                # cv2.imshow('all', output_image1)
                # cv2.waitKey(0)

                # print("normboxes", normboxes)
                # print("1 box", normboxes[0])

                # filtered_imgage
                # output_image = display.show_filted_frame(boxes, self.opt["classes"]) #unblock

                # print("x1", boxes[0][0][0])
                # print("x2", boxes[0][0][2])
                # print("y1", boxes[0][0][1])
                # print("y2", boxes[0][0][3])
                # print(self.opt["classes"])
                # out.add_frame(output_image)

                # display.show_switch(boxes,self.opt["classes"])

                # output_image = []
                # # rail_lines = []
                # for i in range(hn):
                #     # boxes[i] = cut_boxes(boxes[i], height_line, crop_value*2)
                #     #boxes[i] = cut_boxes(boxes[i], height_line, image.shape[3])
                #     im_n = np.array(self.opt['loader'].get_nn_input()[i].permute(1,2,0).cpu().numpy()*255, dtype = "uint8")
                #     # im_n = np.array(input_image[i].permute(1,2,0).cpu().numpy()*255, dtype = "uint8")
                #     # im_n = input_image[i].permute(1,2,0).cpu().numpy()
                #     im_n = np.ascontiguousarray(im_n)
                #     # display.display_line(im_n,boxes[i])
                #     output_image.append(display.display_line(im_n, boxes[i],self.opt["classes"] , height_line=self.opt["loader"].get_nn_height_line()[i]))
                # temp = np.array(image[0,:,self.opt['loader'].get_nn_border():image.shape[2]].permute(1,2,0).cpu().numpy()*255, dtype = "uint8")
                # temp = np.ascontiguousarray(temp)
                # output_image.append(temp)
                # cv2.imshow('display',np.vstack(output_image))
                # cv2.waitKey(0)
                # cv2.destroyWindow('display')

                # opt_alg = {"frames":frame, "boxes":boxes, "rail_lines": rail_lines}
                # opt_alg = {"frames": frame, "boxes": boxes}
                # opt_alg = detect_pipeline.update(opt_alg)
                # На данный момент реализованы алгоритмы определения событий пересечения линии и
                # вхождения в область.
                # Не реализованы классы преобразования событий в подсчет или отправки тревоги
            work_time.append((time.time() - t1) * 1000)

            # cv2.putText(im, str(round((time.time() - t1)*1000, 2)), (150, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)
            # if not display.display(detect_pipeline, opt_alg):
            #     break
        # out.save_video()
        '''
        for tr in sort.trackers:
            p = track_filter_extend(tr, calc_line, self.opt["calc_point"], True)
            self.ret_opt["people_in"] += p[0]
            self.ret_opt["people_out"] += p[1]
        '''
        # ----------------------------------------------------------------------------------------------------------
        self.ret_opt["average_model_time"] = np.mean(work_time)
        self.ret_opt["working_time"] = round((time.time() - self.ret_opt["working_time"]), 2)
        self.ret_opt["frame_counter"] = self.ret_opt["frame_counter"]
        # self.ret_opt["people_in"], self.ret_opt["people_out"] = self.ret_opt["people_in"], self.ret_opt["people_out"]

        print("End test on video %s" % self.opt["VideoFile"])
        return True


def save_resault(opt, ret_opt):
    out_string = "Входные параметры\n"
    op = vars(opt)
    for o in op:
        out_string += str(o) + '= ' + str(op[o]) + '\n'

    out_string += '\n' + 'Выходные параметры\n'
    out_string += ("people in: " + str(ret_opt["people_in"]) + "\n" +
                   "people out: " + str(ret_opt["people_out"]) + "\n" +
                   "Время работы: " + str(ret_opt["working_time"]) + "\n" +
                   "Среднее время обработки одного батча: " + str(ret_opt["average_model_time"]) + "\n" +
                   "Количество кадров: " + str(ret_opt["frame_counter"]) + "\n\n")
    if opt["save_timecode"]:
        out_string += print_table(ret_opt)

    # with open(opt["checkpoint"][:-4] + '.txt', 'w') as f:
    with open('out.txt', 'w') as f:
        f.write(out_string)

    return out_string


def print_table(ret_opt):
    from prettytable import PrettyTable
    table = PrettyTable()

    people_table = []
    time_table = []

    head_table = ["people", "time"]
    people_table = []  # ["people %d line"%(j+1)]
    time_table = []  # ["time %d line"%(j+1)]
    for people, people_time in ret_opt["timecode"]:
        people_table.append(people)
        time_table.append(time.strftime("%H:%M:%S", time.gmtime(people_time)))
    table.add_column(head_table[0], people_table)
    table.add_column(head_table[1], time_table)

    return str(table)


if __name__ == '__main__':
    opt = load_json("./presets/local/teplovoz_railway.json")  # teplovoz_switches.json
    # opt = load_json("./presets/local/teplovoz.json")

    # opt = load_json("./presets/local/Olga_teplovoz_test.json")
    # opt = load_json("./presets/local/scena.json")
    # opt = load_json("./presets/local/preset_classes_cam26_2line_local.json")
    # opt = load_json(sys.argv[1])

    # opt["height"] = 320
    # opt["width"] = 384
    # opt["checkpoint"] = "./weights/last_t1.pt"

    # opt["VideoFile"] = "../../video/reference/Hovanka/237_cam/Сентябрь/08.09.2020/237 2020-09-08 18-29-59_670.mp4"
    # opt["VideoFile"] = "rtsp://192.168.4.103:554/1?stream_id=1.2"
    # opt["VideoFile"] = "/home/evgeny/work/project/lost_thing/ТЕСТ/Движение в запрещенном направлении/UMD_2_TOP.avi"

    opt["VideoFile"] = "video_teplovoz/1.mp4"
    # opt["VideoFile"] = "video_teplovoz/1.mp4"
    # opt["VideoFile"] = "video_teplovoz/123.mp4"
    # opt["VideoFile"] = "video_teplovoz/1.mp4"

    # opt["VideoFile"] = "video_teplovoz/frontal1.mp4"
    # "/home/evgeny/work/project/lost_thing/ТЕСТ/Проход в запрещенную зону/CZ_LO_3_SIDE.avi",
    # opt["VideoFile"] = "/home/evgeny/work/project/video/teplovoz/nuc6/video/18/16/20200916-165800_1280x720_nuc6_cam18_motion.avi"

    task = Task(opt)
    ret = task.start()
    print(task.status())

    # out_string = save_resault(opt, ret_opt)
    # print(out_string)
