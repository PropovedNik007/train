
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from sklearn.decomposition import PCA
from numpy import linalg as LA


distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

def select_device(gpu_id):
    str_cuda_swap = 'cuda:%d' % gpu_id 
    device = torch.device(str_cuda_swap if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)
    return device

def cut_boxes(box_location, height, width):
    box_location[:, [0,2]] = torch.clamp(box_location[:, [0,2]], 0, width)
    box_location[:, [1,3]] = torch.clamp(box_location[:, [1,3]], 0, height)

    
    # box_location[0, -1] = torch.clamp(box_location[0, -1], 0, width)
    # box_location[2, -1] = torch.clamp(box_location[2, -1], 0, width)

    # box_location[1, -1] = torch.clamp(box_location[-1, 1], 0, height)
    # box_location[3, -1] = torch.clamp(box_location[3, -1], 0, height)

    return box_location

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union
    
def cut_intersect1(pred, uoi = 0.9):
    if pred.shape[0] < 2: return pred
    p = pred[:,:4].to('cpu')
    s1 = find_intersection(p,p)
    d = torch.ones(s1.shape[0]) @ (s1*torch.eye(s1.shape[0]))
    s = s1 / d.unsqueeze(0).T
    box = []
    s, _ = s.sort()
    for i, t in enumerate(pred):
        if not (s[i][-2] < uoi):
            continue
        box.append(pred[i].unsqueeze(0))
    return torch.empty(0, 6) if len(box) == 0 else torch.cat(box)
    
def cut_intersect(pred, uoi = 0.75):
    if pred.shape[0] < 2: return pred
    p = pred[:,:4].to('cpu')
    s1 = find_intersection(p,p)
    d = torch.ones(s1.shape[0]) @ (s1*torch.eye(s1.shape[0]))
    s = s1 / d.unsqueeze(0).T

    index = list(range(s.shape[0]))

    n = len(index)
    for i in range(n):
        for j in range(i+1, s.shape[1]):
            if s[i,j] > s[j,i]:
                if s[i,j] > uoi:
                    if i in index:
                        index.pop(index.index(i))
            else:
                if s[j,i] > uoi: 
                    if j in index:
                        index.pop(index.index(j))
    return pred[index]

def mkdir_children(path):
    if path.exists():
        return
    else:
        mkdir_children(path.parent)
        path.mkdir
        # threshold = 200
def get_rail_lines(image_boxes, threshold = 0.034): 
    res = list()
    for image_box in image_boxes:
        gray_patch = cv2.cvtColor(image_box, cv2.COLOR_BGR2GRAY)
        # threshold = np.mean(gray_patch)
        blur1 = cv2.GaussianBlur(gray_patch, (7, 7), 7)
        sobelx2 = cv2.Sobel(blur1, cv2.CV_32F, 1, 0, ksize=3)
        abs_sobel32f = np.absolute(sobelx2)
        sobel_x = np.float32(abs_sobel32f)
        threshold = np.average(sobel_x)

        # _, tres = cv2.threshold(sobel_x, mean_color, 255, cv2.THRESH_BINARY)

        dots = list()
        for i,it in enumerate(sobel_x):
            for i2,it2 in enumerate(it):
                if it2 > threshold:
                    dots.append((i,(sobel_x.shape[1] - i2 - 1)))

        dots = np.array(dots)
        A = np.vstack([dots[:, 0], np.ones(len(dots[:, 0]))]).T
        m, c = np.linalg.lstsq(A, dots[:,1], rcond=None)[0]
        # main_vector = np.array((m*dots[:, 0] + c, dots[:, 0] ))
     
        
        # main_line = [0, image_box.shape[0]-1], [image_box.shape[1] - int(c), image_box.shape[1] -int((image_box.shape[0]-1)*m+c)]
        # res.append([main_line[0][0],main_line[1][0],main_line[1][0],main_line[1][1]])
        res.append([
            0,                                                  # x1
            image_box.shape[1] - int(c),                        # y1
            image_box.shape[0]-1,                               # x2
            image_box.shape[1] -int((image_box.shape[0]-1)*m+c) # y2
        ])
    return np.array(res)




def get_rail_lines_example(image_boxes):
    res = list()
    for image_box in image_boxes:
        image_box = np.uint8(image_box*255)
        gray_patch = cv2.cvtColor(image_box, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray_patch)
        # plt.show()
        # blur1 = cv2.GaussianBlur(gray_patch, (7, 7), 7)
        # plt.imshow(blur1)
        # plt.show()
        # sobelx2 = cv2.Sobel(blur1, cv2.CV_32F, 1, 0, ksize=3)
        # plt.imshow(sobelx2)
        # plt.show()
        # abs_sobel32f = np.absolute(sobelx2)
        # sobel_x = np.float32(abs_sobel32f)
        sobel_x = np.float32(gray_patch)
        mean_color = np.average(sobel_x)

        # _, tres = cv2.threshold(sobel_x, mean_color, 255, cv2.THRESH_BINARY)

        dots = list()
        for i,it in enumerate(sobel_x):
            for i2,it2 in enumerate(it):
                if it2 >= mean_color:
                    dots.append((i2,(i), it2))
        dots.sort(key = lambda x:x[2], reverse = True)
        max_len = int(image_box.shape[0]*image_box.shape[1]*0.2)
        len_d = max_len if len(dots) > max_len else len(dots)
        dots = np.array(dots[:len_d])
        print(dots.shape)

        # plt.scatter(dots[:, 0], dots[:, 1])
        A = np.vstack([dots[:, 0], np.ones(len(dots[:, 0]))]).T
        m, c = np.linalg.lstsq(A, dots[:,1], rcond=None)[0]
        print(m,c)
        # plt.scatter(dots[:, 0], dots[:, 1])

        # plt.plot((0,image.shape[1]-1),(c,(image.shape[1]-1)*m+c), color='red')

        # plt.plot((-m/c, (image.shape[0]-c)/m),(0,image.shape[0]), color='red')

        # plt.scatter(dots[:, 0], dots[:, 1])

        # plt.plot([0, 20],[int(c),int(20*m+c)], color='red')
        
        # plt.plot([0, image_box.shape[1]-1],[int(c),int((image_box.shape[1]-1)*m+c)], color='red')
        # plt.show()
       
        # res.append(x1,x2)
        res.append([
            int(c),                        # y1
            0,                                                  # x1
            int((image_box.shape[1]-1)*m+c), # y2
            image_box.shape[1]-1,                               # x2
        ])
    return np.array(res)


def get_global_line (box, line):
    
    return [box[0]+line[0],box[1]+line[1],box[0]+line[2],box[1]+line[3]]



