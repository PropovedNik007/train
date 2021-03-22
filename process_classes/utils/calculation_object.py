
class calculation_object(object):
    count = 0
    def __init__(self):# object_info объект имеющий всю информацию для создания конкретного типа объекта подсчета
        self.id = calculation_object.count
        calculation_object.count += 1

    def update(self, opt_alg):
        pass
    def get_displayed(self, displayed = {}):
        pass