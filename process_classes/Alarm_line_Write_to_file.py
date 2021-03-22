#import numpy as np

import cv2
import math
import numpy as np
from pathlib import Path
from itertools import chain

from process_classes.utils import *
from tracker_utils import mkdir_children

class Alarm_line_Write_to_file(calculation_object):
    def __init__(self, opt, opt_local, parent):#
        super().__init__()
        self.parent = parent
        if parent is None: raise ValueError("Несоответствующее значение parent")
        self.scenario_id = parent.get_scenario_id()
        self.opt_global = opt
        self.opt_local = opt_local
        self.parse_opt()
        
    def update(self, opt_alg):
        alarm_tr = opt_alg["alarm"]
        if len(alarm_tr) > 0:
            self.save_log(alarm_tr)
        return opt_alg

    def get_displayed(self, displayed = {}):#
        return displayed

    def save_log(self, alarm_tr):

        tm_str = ""
        if self.LogFileName.exists():
            with open(self.LogFileName) as fr:
                tm_str = fr.read()

        for alarm in alarm_tr:
            if alarm["scenario id"] != self.scenario_id: continue
            im_file = ""
            if self.SaveEventsImages:
                im_file = Path(self.opt_global["VideoFile"])
                im_file = self.ImagesDir / (str(self.LogFileName.stem) + "_fr" + str(alarm["frames"]) +\
                    "_ms" + str(int(alarm["frames"] * 1000.0 / self.opt_global["fps"])) +\
                    "_tr"+ str(alarm["tracks id"]) + ".png") #(str(im_file.stem) + "_" + str(self.LogFileName.stem) + str(alarm["tracks id"]) + ".png")
                self.save_img(str(im_file), alarm["image"])
            
            tm_str += str(int(alarm["frames"] * 1000.0 / self.opt_global["fps"])) + ";" + str(im_file) + "\n"

        with open(self.LogFileName, 'w') as fw:
            fw.write(tm_str)

    def save_img(self, file, im):
        cv2.imwrite(file,im)

    def parse_opt(self):#
        self.LogFileName = Path(self.opt_local.get("LogFileName", "log.test"))
        delete_file(self.LogFileName)
        with open(self.LogFileName, 'w') as _:
            pass
        self.ImagesDir = Path(self.opt_local.get("ImagesDir", self.LogFileName.parent))
        mkdir_children(self.ImagesDir)
        self.SaveEventsImages = self.opt_local.get("SaveEventsImages", True)