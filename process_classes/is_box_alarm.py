import numpy as np


class is_box_alarm(object):
    def __init__(self, id):
        self.id = id
    
    def is_box_alarm(self, tracks):
        tr_obj = tracks.calc_objects.get(self.id, {})
        #alarm_history = tr_obj.get("alarm_history", [False])
        alarm_history = tr_obj.get("state_history", [0])
        state = False
        if alarm_history[-1] != 0:
            state = True
        return alarm_history[-1]