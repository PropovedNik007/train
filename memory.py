import numpy as np

class Memory(object):
    previous_data = []
    overlap = 0.3
    perspective = []

    x1_perspective = []
    x2_perspective = []
    x1 = [550, 554, 554, 559, 563, 568, 574, 577, 580, 583, 586, 589, 593, 595, 596, 598, 599, 601, 602, 604, 605]
    x2 = [879, 867, 867, 855, 843, 831, 815, 807, 799, 791, 783, 775, 763, 759, 755, 751, 747, 743, 739, 735, 731]

    y_lines = [720, 672, 624, 576, 528, 464, 432, 400, 368, 336, 304, 256, 240, 224, 208, 192, 176, 160, 144, 128, 112]
    # проведена ли уже иициализация пути
    current_rails_detected = False
    # начальный инициализированный путь
    init_box_detected = []
    # инициализированные до этого пути
    previous_init_boxes = []

    previous_frame_boxes = []

    current_class = ''
    init_box_waiter = 0
    # предидущий бокс в кадре
    previous_box = []
    # текущие пути в кадре
    current_frame_rails = []

    between_frame_counter = 0
    between_frame_switch_front = 0

    previous_switch = 0
    switch_right_front = 0
    switch_left_front = 0

    right_crossing = 0
    left_crossing = 0
