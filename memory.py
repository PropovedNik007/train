
class Memory(object):
    between_frame_counter = 0
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
    previous_switch_right_front = 0
    previous_switch_left_front = 0
