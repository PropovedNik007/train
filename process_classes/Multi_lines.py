#from process_classes.utils.process_utils import calculate_centr_box, angle_between_vectors
#from process_classes.utils.calculation_object import calculation_object
from process_classes.utils import *
from itertools import chain
import numpy as np

################################################

class Line(object):
    def __init__(self, crossLine, offset, color = (255,0,0)):# 
        self.lines = []
        self.color = color

        self.line_offseting(crossLine, offset)#пока стоит заглушка инициализирующая линию без переноса
    
    def get_sides(self):#
        return [tuple(self.points[0]), tuple(self.points[1]), self.color]
    
    def line_offseting(self, crossLine, offset):
        self.points = crossLine
        
    def get_accross_extend(self, centr):
        across = np.zeros(2)
        centr = np.array(centr[-2:], dtype="int64")
        crss = angle_between_vectors(self.points, centr)
        across[0] = crss[1]# if self.direction == 2 else (- crss[1])
        if crss[0] != crss[1]:
            crss = angle_between_vectors(centr, self.points, 0)
            across[1] = 1 if (crss[0] * crss[1]) <= 0 else 0
        return across

################################################

class Multi_lines(calculation_object):
    def __init__(self, opt, opt_local, parent):#
        super().__init__()
        self.parent = parent
        if parent is None: raise ValueError("Несоответствующее значение parent")
        self.scenario_id = parent.get_scenario_id()
        self.opt_global = opt
        self.opt_local = opt_local
        
        line = self.parse_opt()
        self.lines = [Line(line, offset) for offset in self.offsets]


        
    def update(self, opt_alg):#
        for tr in chain(opt_alg["tracks"], opt_alg["del_tracks"]):
            op = track_get_objects_info(tr, self.scenario_id)

            centr = calculate_centr_box(tr.history_box[-1], self.calc_point)
            centr_last = op.centr_history[-1] if centr is None else centr
            op.centr_history.append(centr_last)

            across = np.zeros((len(self.lines), 2))
            for i in range(len(self.lines)):
                across[i][0] = angle_between_vectors(self.lines[i].points, centr_last.reshape(1, -1))

            if len(op.centr_history) >= 2:
                centr = np.array(op.centr_history[-2:], dtype="int64")
                across_old =  op.state_history[-1]

                for i in range(len(self.lines)):
                    if across[i][0] != across_old[i][0]:
                        crss = angle_between_vectors(centr, self.lines[i].points, 0)
                        across[i][1] = 1 if (crss[0] * crss[1]) <= 0 else 0
                        
            op.state_history.append(across)
            ###########################################


            '''
            across = np.zeros((len(self.lines), 2))
            if len(op.centr_history) >= 2:
                for i in range(len(self.lines)):
                    across[i] = self.lines[i].get_accross_extend(op.centr_history)
            op.state_history.append(across)
            '''
            track_set_objects_info(tr, self.scenario_id, op)
        return opt_alg

    def get_displayed(self, displayed = {}):#
        sides = []
        for x in self.lines:
            sides.append(x.get_sides())

        parent_sides = displayed.get("sides", [])
        parent_sides.extend(sides)
        displayed["sides"] = parent_sides

        return displayed

    def parse_opt(self):#

        res = np.array([self.opt_global["width_origin"], self.opt_global["height_origin"]])

        calc_point = self.opt_local.get("calc point", "1.0, 0.5")
        self.calc_point = np.array([float(x) for x in calc_point.split(',')])

        offset = self.opt_local.get("Offset", [0.0])
        self.offsets = (np.array(offset) * res[0]).round()

        line = self.opt_local["line"]
        p1 = [float(x) for x in line['p1'].split(',')]
        p2 = [float(x) for x in line['p2'].split(',')]
        p1, p2 = check_points(p1, p2)
        
        summ = ( sum(p1) + sum(p2))/4
        if summ > 1: 
            res = np.array([1, 1])

        return np.array([p1*res,p2*res], dtype="int64")

################################################