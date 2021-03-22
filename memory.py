
class Memory(object):
    current_rails_detected = False
    init_box_detected = []
    previous_init_boxes = []
    current_frame_vector = []
    current_frame_vector_line = []
    current_class = ''
    init_box_waiter = 0
    previous_box = []

    current_frame_rails = []

    between_frame_counter = 0
    between_frame_switch_front = 0

    previous_switch = 0
    previous_switch_right_front = 0
    previous_switch_left_front = 0
