
#from process_classes.utils import *
import numpy as np
import asyncio
import websockets
import time

from process_classes.utils import *

class Send_to_server(calculation_object):
    def __init__(self, opt, opt_local, parent):#
        super().__init__()
        self.parent = parent
        if parent is None: raise ValueError("Несоответствующее значение parent")
        self.scenario_id = parent.get_scenario_id()
        self.opt_global = opt
        self.opt_local = opt_local
        self.parse_opt()
        
    def update(self, opt_alg):
        try:
            asyncio.get_event_loop().run_until_complete(
                self.send_state(opt_alg)
                )
        except:
            print("connection faled")
        return opt_alg

    async def send_state(self, opt_alg):
        async with websockets.connect(self.uri) as websocket:
            await websocket.send("{\n\"gtp\": \"uuid\"\n}")
            msg = await websocket.recv()
            print(msg)

    def get_displayed(self, displayed = {}):#
        return displayed

    def parse_opt(self):#
        self.uri = "ws://" + self.opt_local["host"] + ":" + self.opt_local["port"]
        self.send_delay = self.opt_local.get("send delay", 1.0)