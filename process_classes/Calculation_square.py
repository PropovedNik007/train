
from process_classes.utils import calculation_object

from itertools import chain
import matplotlib.path
#from process_classes.utils.process_utils import calculate_centr_box
#from process_classes.utils.calculation_object import calculation_object

from process_classes.utils import *
import numpy as np

class Square(object):
    def __init__(self, ROI, scaling = 1, color = (255,0,0)):# 
        self.color = color
        self.ROI = self.scaling_square(np.array(ROI), scaling)
        self.sides = []
        for i in range(len(self.ROI)-1):
            self.sides.append([(int(self.ROI[i][0]), int(self.ROI[i][1])), (int(self.ROI[i+1][0]), int(self.ROI[i+1][1])), self.color])
        self.sides.append([(int(self.ROI[-1][0]), int(self.ROI[-1][1])), (int(self.ROI[0][0]), int(self.ROI[0][1])), self.color])

    def intersection(self, points):
        path = matplotlib.path.Path(self.ROI, closed=False)
        return int(path.contains_points(points, radius=1e-9)) * 2 - 1
    
    def get_sides(self):#
        return self.sides.copy()
    
    def scaling_square(self, ROI, scaling):
        centr = np.mean(ROI, axis = 0)
        return np.int32((ROI - centr)*scaling + centr)

################################################

class Calculation_square(calculation_object):
    def __init__(self, opt, opt_local, parent):#
        super().__init__()
        self.parent = parent
        if parent is None: raise ValueError("Несоответствующее значение parent")
        self.scenario_id = parent.get_scenario_id()
        self.opt_global = opt
        self.opt_local = opt_local
        ROI = self.parse_opt()
        self.box = [Square(ROI, scale) for scale in self.scales]
        
    def update(self, opt_alg):
        for tr in chain(opt_alg["tracks"], opt_alg["del_tracks"]):
            op = track_get_objects_info(tr, self.scenario_id)

            centr = calculate_centr_box(tr.history_box[-1], self.calc_point)
            centr = op.centr_history[-1] if centr is None else centr
            op.centr_history.append(centr)

            across = np.zeros((len(self.box), 2))
            for i in range(len(self.box)):
                across[i][0] = self.box[i].intersection(centr.reshape(1, -1))
            if len(op.centr_history) >= 2:
                across_old =  op.state_history[-1]
                for i in range(len(self.box)):
                    #across[i][0] = self.box[i].intersection(centr.reshape(1, -1))
                    across[i][1] = 1 if (across[i][0] * across_old[i][0]) < 0 else 0
            else:
                for i in range(len(self.box)):
                    if across[i][0] == 1:
                        across[i][1] = 1
            op.state_history.append(across)

            track_set_objects_info(tr, self.scenario_id, op)
        return opt_alg

    def get_displayed(self, displayed = {}):#
        sides = []
        for x in self.box:
            sides.extend(x.get_sides())

        parent_sides = displayed.get("sides", [])
        parent_sides.extend(sides)
        displayed["sides"] = parent_sides

        return displayed

    def parse_opt(self):#
        self.scales = self.opt_local.get("Scale", [1])

        calc_point = self.opt_local.get("calc point", "1.0, 0.5")
        self.calc_point = np.array([float(x) for x in calc_point.split(',')])
        
        ROI = self.opt_local.get("ROI", None)
        if ROI == None:
            return np.array([
                [0.0, 0.0],
                [self.opt_global["width_origin"], 0.0],
                [self.opt_global["width_origin"], self.opt_global["height_origin"]],
                [0.0, self.opt_global["height_origin"]]
                ])
        ROI = np.array([[float(x) for x in str_line.split(',')] for str_line in ROI])
        summ = np.mean(ROI)
        if summ > 1: 
            res = np.array([1, 1])
        else:
            res = np.array([self.opt_global["width_origin"], self.opt_global["height_origin"]])
        return ROI*res