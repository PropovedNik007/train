
def area(set_1, set_2): 
    x1 = max(set_1[0], set_2[0])
    y1 = max(set_1[1], set_2[1])
    x2 = min(set_1[2], set_2[2])
    y2 = min(set_1[3], set_2[3])

    interArea = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((set_1[2] - set_1[0]) * (set_1[3] - set_1[1]))
    boxBArea = abs((set_2[2] - set_2[0]) * (set_2[3] - set_2[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou