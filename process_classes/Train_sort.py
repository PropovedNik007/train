import random
import numpy as np
from .common_class.Sort_origin import Sort_origin


class Train_sort(object):
  def __init__(self, opt, opt_local, parent):#, max_age=10, max_distans=50):
    super().__init__()
    """
    Sets key parameters for SORT
    """
    #################################
    self.parent = parent
    if parent is None: raise ValueError("Несоответствующее значение parent")
    self.parent_id = parent.id
    self.opt_local = opt_local
    self.opt_global = opt
    self.parse_opt()
    #################################

    self.set_global_opt()

    self.history = []
    self.sort_org = Sort_origin()

  def set_global_opt(self):
    self.opt_global["max len history"] = self.max_len_history

  def update(self, opt_alg):
    self.sort_org = Sort_origin()
    opt_alg["box from sort"] = opt_alg["boxes"]
    self.history.append(opt_alg["boxes"])

    for boxs in opt_alg["boxes"][::-1]:
      self.sort_org.update(boxs)

    tracks = self.sort_org.trackers.copy()
    tracks.extend(self.sort_org.del_trackers.copy())
    opt_alg["tracks"] = tracks

    if len(self.history) >= self.max_len_history:
        self.history.pop(0)
        
    return opt_alg

  def get_displayed(self, displayed = {}):#
    return displayed

  def parse_opt(self):#
    self.max_len_history = self.opt_local.get("max len history", 20)
    


