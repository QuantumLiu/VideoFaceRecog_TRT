import cv2
import numpy as np
from itertools import product as product
from math import ceil
def pre_processing(batch_img_src,target_size=(640,640)):
    h_target,w_target=target_size
    batch_img_out=np.ones((len(batch_img_src),*target_size,3),dtype='float32')*128
    h_src,w_src=batch_img_src[0].shape[:2]
    r_h,r_w=h_target/h_src,w_target/w_src
    if r_h>r_w:
        w=w_target
        h=int(h_src*r_w)
        x = 0
        y = int((h_target - h) / 2)
    else:
        h=h_target
        w=int(w_src*r_h)
        x = int((w_target - w) / 2)
        y = 0

    for img_src,img_out in zip(batch_img_src,batch_img_out):

        img_re=cv2.resize(img_src,(w,h),interpolation=cv2.INTER_LINEAR)#.astype('float32')
        img_out[y:y+h,x:x+w]=img_re[:,:]
    #cv2.imwrite('pr.jpg',img_out)
    batch_img_out-=[104., 117., 123.]
    batch_img_out=np.ascontiguousarray(batch_img_out.transpose(0,3,1,2))

    return batch_img_out

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[..., :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[..., 2:] * variances[1])), -1)
    #print(boxes.shape)
    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    #print(priors.shape,pre.shape)
    landms = np.concatenate((priors[:, :2] +pre[..., :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] +pre[..., 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] +pre[..., 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] +pre[..., 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] +pre[..., 8:10] * variances[0] * priors[:, 2:],
                        ), -1)
    return landms




def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def get_prior_data(input_size):
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(input_size[0]/step), ceil(input_size[1]/step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        m_s = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in m_s:
                s_kx = min_size / input_size[1]
                s_ky = min_size / input_size[0]
                dense_cx = [x * steps[k] / input_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / input_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
    output = np.asarray(anchors).reshape(-1, 4)
    return output

PRIOR_DATA_640=get_prior_data((640,640))

def post_processing(trt_outputs,src_size,input_size=(640,640),confidence_threshold=0.2,nms_threshold=0.4,vis_thres=0.6,keep_top_k=750,top_k=5000):
    loc,landms,conf=trt_outputs
    h_src,w_src=src_size
    h_target,w_target=input_size
    r_h,r_w=h_target/h_src,w_target/w_src
    if r_h>r_w:
        w=w_target
        h=int(h_src*r_w)
        x = 0.
        y = float((h_target - h) / 2)
        r=r_w
    else:
        h=h_target
        w=int(w_src*r_h)
        x = float((w_target - w) / 2)
        y = 0.
        r=r_h
    #print(x,y,r)
    variance=[0.1,0.2]
    prior_data=PRIOR_DATA_640 if input_size==(640,640) else (get_prior_data(input_size))
    batch_boxes = decode(loc, prior_data, variance)
    scale_box=np.asarray([input_size[1], input_size[0]]*2)
    batch_boxes = (batch_boxes * scale_box -[x,y]*2)/r
    #batch_boxes = batch_boxes * scale_box
    #batch_boxes = batch_boxes.cpu().numpy()
    batch_scores = conf[..., 1]
    batch_landms = decode_landm(landms, prior_data, variance)
    scale_lm = np.asarray([input_size[1], input_size[0]]*5)

    #scale1 = scale1.to(device)
    batch_landms = (batch_landms * scale_lm -[x,y]*5)/r
    #batch_landms = batch_landms * scale_lm
    #landms = landms.cpu().numpy()

    # ignore low scores
    batch_results=[]
    for boxes,landms,scores in zip(batch_boxes,batch_landms,batch_scores):
        inds = np.where(scores >confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

    # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

    # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

    # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        rects=dets[...,:4]
        scores=dets[...,4]
        batch_results.append((rects,scores,landms))
    return batch_results

def sub_by_scores(rects,scores,landms,thresh=0.6):
    r_n=[]
    s_n=[]
    l_n=[]
    for rect,score,landm in zip(rects,scores,landms):
        if score>=thresh:
            r_n.append(rect)
            s_n.append(score)
            l_n.append(landm)
    return np.asarray(r_n),np.asarray(s_n),np.asarray(l_n)

