from process_classes.utils import calculation_object

from process_classes import Alarm_line_Write_to_file
from process_classes import Calculation_square
from process_classes import is_box_alarm
from process_classes import Hysteresis_line
from process_classes import Multi_lines
from process_classes import Send_to_server
from process_classes import Scenario
from process_classes import Sort
from process_classes import Train_sort

class Class_pipeline(calculation_object):
    def __init__(self, opt, class_pipeline, parent):# opt объект имеющий всю информацию для создания всех типов объектов подсчета
        super().__init__()
        self.objects = []
        self.parent = parent
        self.id = None if parent is None else parent.id
        for obj in class_pipeline:
            name = obj.get("class", None)
            try:
                self.objects.append(
                    globals()[name].__dict__[name](opt, obj, self)#импортируем из пакета "name" класс "name" и инстанцируем его
                )
            except:
                pass

    def update(self, opt_alg):
        for obj in self.objects:
            opt_alg = obj.update(opt_alg)
        return opt_alg

    def get_displayed(self, displayed = {}):#
        for obj in self.objects:
            displayed = obj.get_displayed(displayed)
        return displayed

    def is_box_alarm(self, tracks):
        state = False
        return state

    def get_scenario_id(self):
        # Возвращает id сценария куда следует добавлять данные дочерним алгоритмам
        return None if self.parent is None else self.parent.get_scenario_id()

    def get_child_id(self):
        # Возвращает id дочерних объектов
        ids = []
        for obj in self.objects:
            ids.append(obj.get_child_id())
        return ids

################################################
