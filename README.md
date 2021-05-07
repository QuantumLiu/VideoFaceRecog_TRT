# VideoFaceRecog_TRT
A TensorRT version of video face recog inference lib  
## Dependencies  
CUDA 10.2  
CUDNN 8.0.3  
TensorRT 7.1  
Pytorch 1.4  
Tensorflow 2.0  
Opencv 4.5  
Numpy 1.19  
## Note  
convert.py: convert several .onnx models to trt engine models.  
test_arcface.py: run ArcFace test, predict feature embedding and do compare  
test_retinaface_trt.py: run RetinaFace test, detect faces from image.  
