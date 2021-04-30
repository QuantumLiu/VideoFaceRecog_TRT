import cv2
import time
import numpy as np
from tqdm import tqdm

from models.retinaface.retinaface_res50 import ResNet50RetinaFace
from models.retinaface.utils import pre_processing,post_processing
#from utils.pathes import RESNET50_365_MODEL_PATH_B4,RESNET50_365_MODEL_PATH_B1,RESNET50_365_MODEL_PATH_B8
#from models.yolo.utils import load_class_names,plot_boxes_cv2

batch_size=8

engine_path="/root/pyprojects/VideoFaceRecogInfer/datas/weights/res50retianface_8.engine"
res50=ResNet50RetinaFace(engine_path,batch_size=batch_size)

img=cv2.imread("/root/pyprojects/VideoFaceRecogInfer/datas/imgs/dsns_res_frame.png")
img_c=img.copy()
batch_img_src = np.expand_dims(img, axis=0).repeat(batch_size,axis=0)


for _ in tqdm(range(100)):
    batch_img_in=pre_processing(batch_img_src)
print(batch_img_in.shape)
trt_outputs=res50.predict(batch_img_in)

t_s=time.time()
for _ in tqdm(range(100)):
    trt_outputs=res50.predict(batch_img_in)
t_e=time.time()
time_batch=(t_e-t_s)/100
time_frame=time_batch/batch_size
# loc, land, conf =trt_outputs
src_size=img.shape[:2]
t_s=time.time()
for _ in tqdm(range(100)):
    batch_results = post_processing( trt_outputs,src_size)
t_e=time.time()
time_post=(t_e-t_s)/100
time_post_frame=time_post/batch_size

result=batch_results[-1]
rects,scores,landms=result
print('trt_outputs shape:',trt_outputs[0].shape,'\nNb of faces:{}'.format(len(rects)))
#img=cv2.convertScaleAbs( batch_img_in[0].transpose(1,2,0)+[104, 117, 123])
for rect,score,landm in zip(rects,scores,landms):
    if score<0.5:
        continue
    rect = list(map(int, rect))
    landm = list(map(int, landm))
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
    cx = rect[0]
    cy = rect[1] + 12
    cv2.putText(img, "{:.4f}".format(score), (cx, cy),\
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.circle(img, (landm[0], landm[1]), 1, (0, 0, 255), 4)
    cv2.circle(img, (landm[2], landm[3]), 1, (0, 255, 255), 4)
    cv2.circle(img, (landm[4], landm[5]), 1, (255, 0, 255), 4)
    cv2.circle(img, (landm[6], landm[7]), 1, (0, 255, 0), 4)
    cv2.circle(img, (landm[8], landm[9]), 1, (255, 0, 0), 4)
name = "/root/pyprojects/VideoFaceRecogInfer/test.jpg"
cv2.imwrite(name, img)
#print('Result:\n{}'.format('\n'.join([name+' '*3+'{:4f}'.format(prob) for name,prob in zip(top_names[0],top_probs[0])])))
print('Time cost for batchsize {} :{:6f}  \nper frame : {:6f}\nFPS:{:2f}'.format(batch_size,time_batch,time_frame,1/time_frame))
print('Time cost post processing :{:6f}  \nper frame : {:6f}\nFPS:{:2f}'.format(time_post,time_post_frame,1/time_post_frame))