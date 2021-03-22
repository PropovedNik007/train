#import numpy as np
import cv2
import math
from itertools import chain
import numpy as np

from process_classes.utils import *

class Hysteresis_line(calculation_object):
    def __init__(self, opt, opt_local, parent):#
        super().__init__()
        self.parent = parent
        if parent is None: raise ValueError("Несоответствующее значение parent")
        self.scenario_id = parent.get_scenario_id()
        self.opt_global = opt
        self.opt_local = opt_local
        self.parse_opt()
        
    def update(self, opt_alg):
        opt_alg = self.update_tracks(opt_alg, False)
        #opt_alg = self.update_tracks(opt_alg, True)
        return opt_alg
        
    def update_tracks(self, opt_alg, is_del = False):
        if is_del == False: opt_alg["alarm"] = []
        #tracks = opt_alg["del_tracks"] if is_del else opt_alg["tracks"]
        tracks = chain(opt_alg["tracks"], opt_alg["del_tracks"])
        for tr in tracks:
            op = track_get_objects_info(tr, self.scenario_id)

            hysteresis_line_vars = op.tr_obj.get(self.id, {})
            alarm_history = hysteresis_line_vars.get("alarm_history", [])
            alarm_tr = hysteresis_line_vars.get("alarm", [])

            if op.state_history[-1][0][1] == 0: # если нет пересечения (используется при задержке)
                if len(alarm_history) == 0:
                    continue
                if len(alarm_history) >= self.delay or is_del:
                    if alarm_tr[-1]["direction"] == self.direction or self.direction == 3:
                        opt_alg["alarm"].append(alarm_tr[0])
                        print(int(opt_alg["frames"]*1000.0 /self.opt_global["fps"]), (opt_alg["frames"]), " alarm", alarm_tr[0]["direction"])
                    alarm_history, alarm_tr = [], []
                else:
                    if not (tr.history_box[-1] is None):
                        alarm_history.append(op.state_history[-1][0])# add to history
            '''
            if op.state_history[-1][0][1] == 0: # если нет пересечения (используется при задержке)
                if len(alarm_history) == 0:
                    continue
                if len(alarm_history) >= self.delay or is_del:
                    opt_alg["alarm"].append(alarm_tr[0])
                    print(int(opt_alg["frames"]*1000.0 /self.opt_global["fps"]), (opt_alg["frames"]), " alarm", alarm_tr[0]["direction"])
                    alarm_history, alarm_tr = [], []
                else:
                    if not (tr.history_box[-1] is None):
                        alarm_history.append(op.state_history[-1][0])# add to history
            '''
            if op.state_history[-1][0][1] == 1: # если есть пересечение
                direction = 1 if op.state_history[-1][0][0] == 1 else 2
                direction_old = alarm_tr[-1]["direction"]  if len(alarm_history) > 0 else direction

                if direction_old == direction:
                    alarm = {
                        "scenario id": self.scenario_id,
                        "tracks id": tr.id,
                        "frames":    opt_alg["frames"],
                        "direction": direction,
                        "image": self.get_image(tr)
                    }
                    
                    if self.delay == 0 or is_del:
                        if alarm["direction"] == self.direction or self.direction == 3:
                            opt_alg["alarm"].append(alarm)
                            print(int(opt_alg["frames"]*1000.0 /self.opt_global["fps"]), (opt_alg["frames"]), " alarm", alarm["direction"])
                    else:
                        alarm_tr.append(alarm)
                        alarm_history.append(op.state_history[-1][0])# add to history
                else:
                    alarm_history, alarm_tr = [], []
               
                '''
                if alarm["direction"] == self.direction or self.direction == 3:
                    if self.delay == 0 or is_del:
                        opt_alg["alarm"].append(alarm)
                        print(int(opt_alg["frames"]*1000.0 /self.opt_global["fps"]), (opt_alg["frames"]), " alarm", alarm["direction"])
                    else:
                        alarm_tr.append(alarm)
                        alarm_history.append(op.state_history[-1][0])# add to history
                else:
                    alarm_history, alarm_tr = [], [] # надо додумать механизм сброса тревоги. сейчас стоит наивный алгоритм
                '''
            hysteresis_line_vars["alarm_history"] = alarm_history
            hysteresis_line_vars["alarm"] = alarm_tr
            op.tr_obj[self.id] = hysteresis_line_vars

            track_set_objects_info(tr, self.scenario_id, op)
        return opt_alg

    def get_image(self, tr):#
        im = np.copy(self.opt_global["loader"].get_image_origin())
        displayed = self.parent.get_displayed({})
        lines = displayed.get("sides", [])
        for line in lines:
            cv2.line(im, line[0], line[1], line[2], thickness=2) 
            
        display_track(im, tr, (0, 255, 255))

        return im

    def get_displayed(self, displayed = {}):#
        return displayed

    def parse_opt(self):#
        self.delay = math.ceil(self.opt_local.get("time delay", 0) * self.opt_global["fps"])
        self.direction = self.opt_local.get("direction", 3)