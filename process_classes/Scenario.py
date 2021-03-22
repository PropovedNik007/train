from process_classes.utils import calculation_object
import process_classes.Class_pipeline 

class Scenario(calculation_object):
    def __init__(self, opt_global, opt_local, parent):#
        super().__init__()
        self.parent_id = parent.id
        self.opt_global = opt_global
        self.opt_local = opt_local
        self.parse_opt()
        self.pipeline = process_classes.Class_pipeline.Class_pipeline(opt_global, self.opt_local.get("Class pipeline", None), self)
        
        
    def update(self, opt_alg):
        opt_alg = self.pipeline.update(opt_alg)
        return opt_alg

    def get_displayed(self, displayed = {}):#
        return self.pipeline.get_displayed(displayed) if self.show_track else {}
    
    def parse_opt(self):#
        self.Scenario_name = self.opt_local.get("Scenario name", str(self.id))
        self.show_track = self.opt_local.get("Show track", True)

    def get_scenario_id(self):
        # Возвращает id сценария куда следует добавлять данные дочерним алгоритмам
        return self.id