"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import random
import numpy as np
from filterpy.kalman import KalmanFilter as KF

#np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def near_batch(bb_test, bb_gt):      

  CurrentCentrBox =(bb_test[:,:2] + bb_test[:,2:4])/2      
  PrevisionCentrBox =(bb_gt[:,:2] + bb_gt[:,2:4])/2       

  mm = np.zeros((len(CurrentCentrBox), len(PrevisionCentrBox)))

  for i, cb in enumerate(CurrentCentrBox):
      for j, pb in enumerate(PrevisionCentrBox):
          m = np.power(np.sum(np.power(cb.astype(np.float32) - pb.astype(np.float32), 2)), 0.5)
          mm[i,j] =  m
  return mm

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanFilter(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KF(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self, count):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    return convert_x_to_bbox(self.kf.x)

class LinearFilter(object):
  def __init__(self,bbox):
    self.x = [convert_bbox_to_z(bbox)]

  def update(self, bbox):
    x = convert_bbox_to_z(bbox)
    self.x.append(x)
    if len(self.x) > 2: self.x.pop(0)

  def predict(self, count):
    count = 1 #+ count*0.2
    if len(self.x) < 2: return convert_x_to_bbox(self.x[0])
    x0 = self.x[1][0] + (self.x[1][0] - self.x[0][0]) * count
    x1 = self.x[1][1] + (self.x[1][1] - self.x[0][1]) * count
    x2 = max(self.x[1][2],self.x[0][2])
    x3 = self.x[1][3] if self.x[1][2] > self.x[0][2] else self.x[0][3]
    
    return convert_x_to_bbox(np.array([x0,x1,x2,x3]))

class LinearMomentiumFilter(object):
  def __init__(self,bbox):
    self.x = [convert_bbox_to_z(bbox)]
    self.alfa = 0.95
    self.step = np.array([0.0, 0.0])

  def update(self, bbox):
    x = convert_bbox_to_z(bbox)
    if len(self.x) < 2: 
      self.x.append(x)
    else:
      self.x[-1] = x

  def predict(self, count):
    if len(self.x) < 2: return convert_x_to_bbox(self.x[0])
    x = np.zeros(4)
    step = (self.x[1][:,0] - self.x[0][:,0])[:2]

    if self.step[0] * self.step[1] == 0:
      self.step = step
    else:
      self.step = step * (1 - self.alfa) + self.step * self.alfa

    x[:2] = self.x[1][:2,0] + step
    x[2:4] = self.x[1][2:4,0] if self.x[1][2][0] > self.x[0][2][0] else self.x[0][2:4,0]

    self.x.append(x.reshape((4,1)))
    if len(self.x) > 2: self.x.pop(0)
    
    return convert_x_to_bbox(x)
  

        
class LinearBoxTracker(object):
  count = 0
  def __init__(self,bbox, frames):
    # self.filter = LinearFilter(bbox[:4])
    self.filter = LinearMomentiumFilter(bbox[:4])
    #self.filter = KalmanFilter(bbox[:4]) 
    
    self.time_since_update = 0
    self.id = LinearBoxTracker.count
    LinearBoxTracker.count += 1
    self.start_frame = frames # номер кадра в котором началось отслеживание
    self.history = []
    self.history_box = [bbox]
    self.calc_objects = dict()# Словарь для хранения информации по объектам подсчета
    #Ключем словаря является уникальный ID сценария. Значением являются словари:
    # state_history - история состояний объекта в с момента начала отслеживания
    # centr_history - история положений точки подсчета объекта в с момента начала отслеживания
    #
    self.history_im = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.color_box = [random.randint(0, 255) for _ in range(3)]
  
  def add_im(self,im):
    if len(self.history_box) > len(self.history_im):
      self.history_im.append(im)

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    #self.history = []
    self.history_box[-1] = bbox
    self.hits += 1
    self.hit_streak += 1
    self.filter.update(bbox[:4])

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.history = []

    self.age += 1
    self.history_box.append(None)
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    x = self.filter.predict(self.time_since_update)
    self.history.append(x)
    return self.history[-1]

def associate_near_detections_to_trackers(detections,trackers,max_distans = 50):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  # iou_matrix = iou_batch(detections, trackers)
  iou_matrix = near_batch(detections, trackers)
  

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix < max_distans).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]>max_distans):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, opt, opt_local, parent):#, max_age=10, max_distans=50):
    super().__init__()
    """
    Sets key parameters for SORT
    """
    self.parent = parent
    if parent is None: raise ValueError("Несоответствующее значение parent")
    self.parent_id = parent.id
    self.opt_local = opt_local
    self.opt_global = opt
    self.parse_opt()

    self.trackers = []

    self.set_global_opt()

  def set_global_opt(self):
    self.opt_global["max distans"] = self.max_distans

  def update(self, opt_alg):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets = opt_alg["boxes"]
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), dets.shape[1]))
    to_del = []
    del_track = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
      else:
        trk[:4] = pos
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      del_track.append(self.trackers.pop(t))
    matched, unmatched_dets, unmatched_trks = associate_near_detections_to_trackers(dets, trks, self.max_distans)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])
      
    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = LinearBoxTracker(dets[i, :], opt_alg["frames"])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          del_track.append(self.trackers.pop(i))
    opt_alg["tracks"], opt_alg["del_tracks"] = self.trackers, del_track
    return opt_alg

  def get_displayed(self, displayed = {}):#
    return displayed

  def parse_opt(self):#
    self.max_age = self.opt_local.get("max age", 20)
    self.max_distans = self.opt_local.get("max distans", 50)