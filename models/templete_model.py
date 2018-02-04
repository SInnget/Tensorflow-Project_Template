from base.base_model import BaseModel
import tensorflow as tf


class TempleteModel(BaseModel):
    def __init__(self, config):
        super(TempleteModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        pass

    def init_saver(self):
        #here you initalize the tensorflow saver that will be used in saving the checkpoints.
        pass