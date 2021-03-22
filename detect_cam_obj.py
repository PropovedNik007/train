import argparse
import time
from pathlib import Path

import os
import sys
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


pth = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(str(pth))
from yolov5 import *

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path, bbox_iou # check_requirements
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized


#python3 detect_cam_obj.py --weights=/media/user3070/data/yolov5_olga_test/runs/train/exp74/weights/best.pt --source=/media/user3070/data/Teplovoz_video/vid --img-size=608

def sort_pred(prediction, iou_tr):

    sorted_pred = []
    for k in range(len(prediction[0])):
        for j in range(len(prediction[0])):
            if (k + j + 1) == len(prediction[0]):
                break
            iou = bbox_iou(prediction[0][k], prediction[0][k+j+1], x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9)
            if iou >= iou_tr:
                if prediction[0][k][4] > prediction[0][k+j+1][4]:
                    sorted_pred.append(k+j+1)
                else:
                    sorted_pred.append(k)

    prediction = prediction[0].cpu().detach().numpy()
    prediction = np.delete(prediction, sorted_pred, 0)
    prediction = torch.tensor([prediction])

    return prediction   
                                                         


def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
 
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() 
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        pred = sort_pred(pred, iou_tr=0.6) # отсекает пересекающиеся боксы с меньшей вероятностью 
        
        for i, det in enumerate(pred):
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            for k in range(det.size(0)):
                box = det[k].tolist()
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                text = names[int(box[5])] + ' ' + str(round(box[4], 4))
                print('classes:', names[int(box[5])])
                cv2.rectangle(im0s, (x1, y1), (x2, y2), colors[int(box[5])], 2)
                cv2.putText(im0s, text, tuple([x1, y2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
            
            cv2.imshow('Image', im0s)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()