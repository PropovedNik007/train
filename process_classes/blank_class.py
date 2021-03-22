from itertools import chain

from process_classes.utils import *

class Track_class(calculation_object):
    def __init__(self, opt, opt_local, parent):#
        self.parent = parent
        if parent is None: raise ValueError("Несоответствующее значение parent")
        self.scenario_id = parent.get_scenario_id()
        self.opt_global = opt
        self.opt_local = opt_local
        self.parse_opt()
        
    def update(self, opt_alg):
        #for tr in chain(opt_alg["tracks"], opt_alg["del_tracks"]):
        pass

    def get_displayed(self, displayed = {}):#
        return displayed

    def parse_opt(self):#
        pass