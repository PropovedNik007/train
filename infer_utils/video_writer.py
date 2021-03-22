import cv2

class Video_holder():
    def __init__(self,name,width,height):
        self.video = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'MPEG'),20,(width,height))
    def add_frame(self,frame):
        self.video.write(frame)
    def save_video(self):
        self.video.release()
