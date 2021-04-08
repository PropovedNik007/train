import numpy as np

class InputPlug(object):
    height_line = [16, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 96, 96, 96]

    CropCoors0 = np.float32([624, 576, 528, 480, 432, 400, 368, 336, 304, 272, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80,
                  64, 48, 32, 16, 8, 0])
    CropCoors1 = np.float32([720, 672, 624, 576, 528, 464, 432, 400, 368, 336, 304, 256, 240, 224, 208, 192, 176, 160, 144, 128,
                  112, 96, 80, 64, 48, 24, 16])
    CropCoors = np.column_stack((CropCoors0, CropCoors1))
    # CropCoors = np.column_stack((CropCoors0, CropCoors1))[::-1]

    FrameWidth = 1280
    FrameHeight = 720
    FrameNumber = 666

    # rail_boxes, crop_coors, frame_width, frame_height

    CropCoors = CropCoors / FrameHeight

    # def input_translate(self):
    #     InputPlug.CropCoors = np.int16(InputPlug.CropCoors * InputPlug.FrameHeight)
