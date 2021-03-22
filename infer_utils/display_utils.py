import numpy as np

def area(set_1, set_2): 
    xA = max(set_1[0], set_2[0])
    yA = max(set_1[1], set_2[1])
    xB = min(set_1[2], set_2[2])
    yB = min(set_1[3], set_2[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((set_1[2] - set_1[0]) * (set_1[3] - set_1[1]))
    boxBArea = abs((set_2[2] - set_2[0]) * (set_2[3] - set_2[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_result_box_coor(group):
	if len(group) < 3:
		return False
	coors = [group[-1][0], group[0][1], group[-1][2], group[-1][3]]
	return coors


def get_result_switch(switches):

	ious = list(map(area, switches, switches[1:]))
	indexes = [i for i, x in enumerate(ious) if x == 0]

	if len(indexes) != 0:
		switches_group = []
		switches_group.append(switches[:indexes[0]+1])
		for j in range(len(indexes)-1):
			switches_group.append(switches[indexes[j]+1:indexes[j+1]+1])
		switches_group.append(switches[indexes[-1]+1:])

		result_boxes = list(map(get_result_box_coor, switches_group))
	else: 
		result_boxes = [get_result_box_coor(switches)]

	return result_boxes