import cv2
import numpy as np
import time
from tracker_utils import distinct_colors
from process_classes.utils import display_box, display_track

class frame_display_multi_line(object):
    def __init__(self, view_cam = False, video_loader = None, delay = 0):
        #delay - Управление режимом вывода. 1 видеорежим, 0 покадровый режим
        self.delay = delay
        self.view_cam = view_cam
        self.video_loader = video_loader
        if video_loader is None:
            self.width_origin = 0
            self.height_origin = 0
            self.view_cam = False
        else:
            self.height_origin, self.width_origin = video_loader.get_origin_resolution()

        if self.width_origin == 0 or self.height_origin == 0: self.view_cam = False

        self.init_constant()
    
    def init_constant(self):
        self.title = "--- Glosav analytics ---"
        self.font            = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale       = 1
        self.thickness       = 2
        self.calc_lines = []
            
        self.org3 = (10, int(self.height_origin-7)) 
        self.color3 = (0, 255, 255)

        if not self.view_cam: return
        self.init_window() 
    
    def init_window(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1280, 720) 

    def set_view_mode(self, state):
        self.view_cam = state
        self.init_constant()

    def display_prediction_point(self, calc_bilder, track, colores = None):
        for tr in track:
            color = tr.color_box if colores is None else colores
            state = calc_bilder.is_box_alarm(tr)
            color = (0, 0, 255) if state else color
            
    

    def display_tracks(self, calc_bilder, track, colores = None):
        for tr in track:
            color = tr.color_box if colores is None else colores
            state = calc_bilder.is_box_alarm(tr)
            color = (0, 0, 255) if state else color
            display_track(self.im, tr, color)     

            if len(tr.history):
                boxe = tr.history[-1][0]
                boxe = np.int32( (boxe[2:4] + boxe[:2]) / 2)
                max_distans = calc_bilder.objects[0].opt_global["max distans"]
                # cv2.circle(self.im, tuple(boxe), max_distans, color, 3)
                # cv2.circle(self.im, tuple(boxe), 3, color, 2)



    def display(self, calc_bilder, opt_alg):#frame_counter, track, del_track):
        if not self.view_cam: return True

        self.set_image()
        self.display_tracks(calc_bilder, opt_alg["tracks"])
        self.display_tracks(calc_bilder, opt_alg["del_tracks"], (0, 255, 0))

        im = cv2.putText(self.im, 'Frame: %d' % opt_alg["frames"], self.org3, self.font, self.fontScale, self.color3, self.thickness, cv2.LINE_AA)
        
        for calc_object in calc_bilder.objects:
            displayed = calc_object.get_displayed({})
            lines = displayed.get("sides", [])
            for line in lines:
                cv2.line(self.im, line[0], line[1], line[2], thickness=2)
        
                
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_AUTOSIZE) != 0:
            self.init_window() 
        cv2.imshow(self.title, im)

        key = cv2.waitKey(self.delay)
        if key & 0xFF == 13 or key & 0xFF == 141: #Если нажали Enter то перейти в видеорежим
            self.delay = 20
        if key & 0xFF == 32: #Если нажали Space то перейти в покадровый режим
            self.delay = 0
        if key & 0xFF == ord('q') or key & 0xFF == 202: #Если нажали Q то выйти
            return False
        return True
            
    def set_image(self):
        if self.video_loader is None or not self.view_cam:
            self.im  = None
        else:
            #self.im  = loader.get_image()
            self.im  = self.video_loader.get_image_origin()