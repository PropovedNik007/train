class Line:
    def __init__(self, box):
        self.y0 = box[1]
        self.y1 = box[3]


class Rails(Line):
    def __init__(self, box, previous, next):
        super().__init__(box)
        self.x0 = box[0]
        self.x1 = box[2]

        self.box_class = box[4]
        self.previous = previous

class CentralRails(Rails):
    pass
