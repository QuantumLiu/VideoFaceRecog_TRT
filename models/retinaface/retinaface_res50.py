import numpy as np
from models.base import BaseNet


try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


class ResNet50RetinaFace(BaseNet):
    def __init__(self,engine_path,image_size=(640,640),batch_size=1):
        super().__init__(engine_path,image_size,batch_size)
    def predict(self,batch_img_in):
        trt_outputs=super().predict(batch_img_in)
        trt_outputs[0] = trt_outputs[0].reshape(self.batch_size,-1,4)
        trt_outputs[1]=trt_outputs[1].reshape(self.batch_size,-1,10)
        trt_outputs[2]=trt_outputs[2].reshape(self.batch_size,-1,2)
        return trt_outputs


