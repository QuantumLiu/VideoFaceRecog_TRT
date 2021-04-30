import sys
import os
import time
import argparse
import numpy as np
import tensorrt as trt
print("TensorRT init")
from trt_backend.engine import load_engine
from trt_backend.engine import cuda, autoinit
from trt_backend.engine import HostDeviceMem, GiB, allocate_buffers
from trt_backend.engine import DEFAULT_TRT_LOGGER,DEFAULT_TRT_RUNTIME

from trt_backend.engine import do_inference,allocate_buffers

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

class BaseNet():
    def __init__(self,engine_path,image_size=(256,256),batch_size=1,channel_first=True):#,num_classes=365):
        self.engine_path=engine_path

        # self.q_in=q_in
        # self.q_out=q_out

        #self.num_classes=num_classes
        self.image_size=image_size
        self.batch_size=batch_size
        self.channel_first=channel_first


        IN_IMAGE_H, IN_IMAGE_W = self.image_size
        if self.channel_first:
            self.binding_shape=(self.batch_size, 3, IN_IMAGE_H, IN_IMAGE_W)
        else:
            self.binding_shape=(self.batch_size, IN_IMAGE_H, IN_IMAGE_W, 3)
        self.engine=None
        self._build_engine()
        self._build_context()
        
        self.buffers = allocate_buffers(self.engine, self.batch_size)
    def _build_engine(self):
        self.engine=load_engine(self.engine_path)
    def _build_context(self):
        self.context=self.engine.create_execution_context()
        # min_shape, opt_shape, max_shape = self.engine.get_profile_shape(self.context.active_optimization_profile,0)
        # print(min_shape, opt_shape, max_shape)
        # print(self.binding_shape)
        result_set_binding=self.context.set_binding_shape(0, self.binding_shape)
        if not result_set_binding:
            Warning("********Set binding ERROR********")
    
    def _infer(self,batch_img_in):
        batch_img_in = np.ascontiguousarray(batch_img_in)
        #print("Shape of the network input: ", batch_img_in.shape)
        inputs, outputs, bindings, stream = self.buffers
        #print('Length of inputs: ', len(inputs))
        inputs[0].host = batch_img_in

        trt_outputs = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        #print('Len of outputs: ', len(trt_outputs))

        #trt_outputs[0] = trt_outputs[0].reshape(self.batch_size,self.num_classes)
        #trt_outputs[1] = trt_outputs[1].reshape(self.batch_size, -1, self.num_classes)
        return trt_outputs
    
    def predict(self,batch_img_in):
        return self._infer(batch_img_in)
