import cv2
import numpy as np
from models.arcface.archface_res50 import ResNet50ArcFace
from models.arcface.utils import l2_norm
engine_path="/root/pyprojects/VideoFaceRecogInfer/datas/weights/arcfaceres50_1.engine"
arcface=ResNet50ArcFace(engine_path,channel_first=False)
img1=cv2.imread('/root/pyprojects/VideoFaceRecogInfer/joey0.ppm')
img2=cv2.imread('/root/pyprojects/VideoFaceRecogInfer/joey1.ppm')
#cv2.imwrite('t.jpg',img1)
img1_in=img1.astype(np.float32)[None,:]/255
img2_in=img2.astype(np.float32)[None,:]/255
emb1=l2_norm(arcface.predict(img1_in)[0])
emb2=l2_norm(arcface.predict(img2_in)[0])
print(emb1.shape,np.linalg.norm(emb1),np.linalg.norm(emb2))
print( np.dot(emb1,emb2),np.allclose(emb1,emb2))
