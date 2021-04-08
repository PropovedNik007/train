import cv2
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class data_loader(object):
    def __init__(self):
        pass
    
    def next_frame(self):#
        pass

    def get_tensor(self):#
        pass
    
    def get_image_origin(self):#
        pass

    def get_image(self):#
        pass

    def get_resolution(self):#
        pass
    
    def get_origin_resolution(self):#
        pass

    def get_frame_count(self):#
        pass

    def get_read_frames(self):
        pass

class video_loader(data_loader):
    """docstring"""
    
    def __init__(self, VideoFile, img_size = (448,544), batch_size = 1, half = True):
        self.half = half
        self.VideoFile = VideoFile
        self.batch_size = int(batch_size)
        assert self.batch_size > 0 
        self.frame_counter = 0
        self.cap = cv2.VideoCapture(VideoFile)
        self.image = None
        self.image_tensor = None
        self.image_origin = None
        self.raw_image_tensor = None

        self.width_origin = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height_origin = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.width = img_size[1]
        self.height = img_size[0]

        # Высоты лент на изображении для формирования входных данных YOLO
        self.height_line = [32,32,32,32,32,64,64,64,96,96,96]
        #self.height_line = self.height_line[::-1]
        #self.height_line = [16,16,16,16,16,32,32,32,32,32,64,64,64]
        #self.height_line = [32,32,32,32,32,32,32,64,64,64,96]
        # Высота лент на входе в YOLO
        # self.height_standart = 64

        self.b = 1

        if not self.cap.get(cv2.CAP_PROP_FPS) \
                    or not self.width_origin  \
                    or not self.height_origin:
            raise Exception("Missing or damaged video file")
    
    def next_frame(self):

        self.image = None               # текущий считанный кадр разрешения [300,300,3]
        self.image_origin = None        # текущий считанный кадр разрешения [576,720,3]
        self.image_tensor = None        # преобразованный в тензор текущий кадр разрешения [3,300,300]
        self.nn_input = []              # Вход для YOLO в формате тензор
        self.top_border = self.height_origin - sum(self.height_line) # Граница с которой режется лента
        self.border = self.top_border

        self.num_image = 0
        input_list = []
        for _ in range(self.batch_size):
            ret, image = self.cap.read()
            
            if ret:
                self.image_origin = image

                self.image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

                self.top_list = []
                self.bottom_list = []
                # for h in self.height_line:
                #     temp = image[self.border:self.border+h] 
                #     self.top_list.append(self.border)
                #     self.bottom_list.append(self.border+h)
                #     # Изменение высоты ленты под размер входа в YOLO
                #     self.nn_input.append(cv2.resize(temp,\
                #         dsize=(self.width,self.get_height_standart(temp.shape[0])),interpolation=cv2.INTER_AREA))
                #     # Сдвиг границы на h
                #     self.nn_input[-1] = self.nn_input[-1][:, :, ::-1].transpose(2, 0, 1)
                #     self.nn_input[-1] = np.ascontiguousarray(self.nn_input[-1])
                #     self.border+=h
                
                #c перекрытием
                for h in self.height_line:
                    for step in list([0, int(h/2)]):
                        if self.border+h+step <= self.height_origin:
                            temp = image[self.border + step:self.border+h+step]   
                            self.top_list.append(self.border + step)
                            self.bottom_list.append(self.border+h+step)
                            # Изменение высоты ленты под размер входа в YOLO
                            self.nn_input.append(cv2.resize(temp,\
                                dsize=(self.width,self.get_height_standart(temp.shape[0])),interpolation=cv2.INTER_AREA))
                            # Сдвиг границы на h
                            self.nn_input[-1] = self.nn_input[-1][:, :, ::-1].transpose(2, 0, 1)
                            self.nn_input[-1] = np.ascontiguousarray(self.nn_input[-1])
                    self.border+=h
                
                # self.nn_input = np.stack(self.nn_input)
                
                img = self.image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                # self.image_tensor = torch.from_numpy(img) #torch.FloatTensor(img)
                for i, it in enumerate(self.nn_input):
                    self.nn_input[i] = torch.from_numpy(self.nn_input[i]).cuda()
                # self.nn_input = torch.from_numpy(self.nn_input)                

                # input_list.append(self.nn_input.unsqueeze(0).cuda())
                # input_list.extend(self.nn_input.unsqueeze(0).cuda())
                self.num_image += 1
            else:
                if self.num_image == 0:
                    return False

        self.frame_counter += self.num_image
        # self.nn_input = torch.cat(input_list)
        for i,it in enumerate(self.nn_input):
            # self.nn_input[i] = # torch.squeeze(self.nn_input)
            self.nn_input[i] = it.half() if self.half else it.float() # fp16/32
            self.nn_input[i] /= 255.0 
      
        
        # for h in self.height_line:
        #     # Вырезается лента по высоте с границы размером в h
        #     temp = self.image_tensor[:, :, self.border:self.border+h] 
        #     # Изменение высоты ленты под размер входа в YOLO
        #     self.nn_input.append(cv2.resize(temp,\
        #         dsize=(self.height_standart,self.width_origin),interpolation=cv2.INTER_AREA))
        #     # Сдвиг границы на h
        #     self.border+=h
        # self.nn_input = torch.cat(self.nn_input,dim=0)

        return True
        
    def get_height_standart(self,img_height):
        if img_height == 32 or img_height==96:
            return 64
        return img_height

    def get_tensor(self):
        return self.image_tensor
    
    def get_image_origin(self):
        return self.image_origin

    def get_image(self):
        return self.image

    def get_resolution(self):
        return self.height, self.width
    
    def get_origin_resolution(self):
        return self.height_origin, self.width_origin

    def get_read_frames(self):
        return self.frame_counter

    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_nn_input(self):
        return self.nn_input
    
    def get_nn_top_border(self):
        return self.top_border
        
    def get_nn_border(self):
        return self.border
    
    def get_nn_height_line(self):
        return self.height_line

    def set_frame(self, frame_counter):
        self.cap.set(1, frame_counter-1)
        self.frame_counter = frame_counter-1
        return

    def get_top_bottom_list(self):
        return self.top_list, self.bottom_list