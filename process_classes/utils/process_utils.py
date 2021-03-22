import cv2
import numpy as np

class operand(object):
    def __init__(self):
        pass
    
def delete_file(file):
    try:
        file.unlink()
    except: pass
    
def calculate_centr_box(boxes, calc_point):
    if boxes is None: return None
    boxe = np.array(boxes)
    w_h = (boxe[:,2:4] - boxe[:,:2])*calc_point
    centr = np.int32(boxe[:, :2] + w_h)
    return centr

def angle_between_vectors(points_line, points, eps = 0.0000003):
    # start, end должны быть типа numpy array
    # points должен быть типа numpy array или приводится к numpy array конструкцией np.array(points)
    #считает знак угла между векторами (points_line[0], points_line[1]) и (points_line[0], points[i])
    vec1 = points - points_line[0]
    vec2 = points_line[1] - points_line[0]
    angle = -(vec1[:, 0]*vec2[1]-vec1[:, 1]*vec2[0]) +  eps# eps добавлено чтобы точки не попадали на линию
    angle = angle / np.abs(angle)
    return angle

def check_points(p1, p2):
    if p1[1] == p2[1]:
        if p2[0] > p1[0]:
            p1, p2 = p2, p1
    elif p2[1] > p1[1]:
        p1, p2 = p2, p1
    return p1, p2


def track_get_objects_info(track, scenario_id):
    op = operand()
    op.tr_obj = track.calc_objects.get(scenario_id, {})
    op.state_history = op.tr_obj.get("state_history", [])
    op.centr_history = op.tr_obj.get("centr_history", [])
    return op

def track_set_objects_info(track, scenario_id, op):
    op.tr_obj["state_history"] = op.state_history
    op.tr_obj["centr_history"] = op.centr_history
    track.calc_objects[scenario_id] = op.tr_obj
        
def display_box(im, box, color, thickness =1):
#     im_n = np.array(im.permute(1,2,0).cpu().numpy()*255, dtype = "int8")
#     im_n = np.ascontiguousarray(im_n)
    cv2.rectangle(im, (int(box[0]), int(box[1])),
        (int(box[2]),int(box[3])), color, thickness =1)
    

def display_track(im, tr, color):
    # for val in tr.calc_objects.values():
    #     centr = val["centr_history"]
    #     for i in range(1, len(centr)):
    #         cv2.line(im, tuple(np.int32(centr[i-1])),
    #             tuple(np.int32(centr[i])), color, thickness=3)

    if tr.history_box[-1] is None: return
    display_box(im, tr.history_box[-1], color)
    text = str(int(tr.id+1))
    cv2.putText(im, text, (int(tr.history_box[-1][0]),
        int(tr.history_box[-1][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)