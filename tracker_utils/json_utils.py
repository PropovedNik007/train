
import json
import numpy as np

def load_jsons(data):
    opt = {}

    opt["view_cam"]      = data.get('view cam', False)         # показывать видеоизображение

    model = data.get('model', {})
    opt["gpu_id"]      = model.get('gpu_id', 0)        # Номер GPU для расчета
    opt["batch_size"]  = model.get('batch size', 1)    # размер батча
    opt["half"]        = model.get('half', True)       # расчет на float point 16, иначе FP32
    opt["checkpoint"]  = model.get('checkpoint', "./weights/yolov5l.pt")   # путь до весов модели

    
    algoritm_setting = data.get('algoritm setting', {})
    opt["min_score"] = algoritm_setting.get('min score', 0.5)                  # порог обнаружения объекта
    opt["overlap"]   = algoritm_setting.get('overlap', 0.6)                      # порог пересечения объектов для алгоритма Non-Maximum Suppression (NMS)
    opt["classes"]   = algoritm_setting.get('classes')
    opt["line heght"]   = algoritm_setting.get('line heght', 64)
    
    work_resolution = data.get('work resolution', {})
    opt["work_height"] = work_resolution.get('height', 320) # Разрешение по высоте
    opt["work_width"]  = work_resolution.get('width', 384)  # Разрешение по ширине
    
    Scena = data.get('Scena', {})
    opt["VideoFile"]   = Scena.get('VideoFile', None) # "rtsp://192.168.4.103:554/20?stream_id=20.2" # путь до видеофайла или rtsp потока
    opt["scenarios"]    = Scena.get('Class pipeline', []) 
    return opt

def load_json(file):
    with open(file, 'r', encoding='utf-8-sig') as read_file:
        data = json.load(read_file)
        return load_jsons(data)