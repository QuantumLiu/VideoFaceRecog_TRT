import os
import uuid
import traceback

import numpy as np
import cv2

from models.arcface.archface_res50 import ResNet50ArcFace
from models.arcface.utils import l2_norm,get_transform_mat,warp_img

from models.retinaface.retinaface_res50 import ResNet50RetinaFace
from models.retinaface.utils import pre_processing,post_processing,sub_by_scores

from utils.pathes import RESNET50_RETINAFACE_MODEL_PATH_B1,RESNET50_ARCFACE_MODEL_PATH_B1
from utils.facial import _parse_info,_sub_feature,_load_info,_save_info,remove_old,save_facial_infos

_ARCFACE=ResNet50ArcFace(RESNET50_ARCFACE_MODEL_PATH_B1)
_RETINAFACE=ResNet50RetinaFace(RESNET50_RETINAFACE_MODEL_PATH_B1)

def worker_dir(path_dir,over_write=False):
    print('Registing dir:{}'.format(path_dir))
    info_path=os.path.join(path_dir,'info.json')
    empty_flag_path=os.path.join(path_dir,'empty.flag')
    nb_imgs=0
    name_celeba,sex_id=_parse_info(path_dir)
    uid=str(hash(name_celeba+str(sex_id)))
    if ((not os.path.exists(info_path)) or over_write):# and not (os.path.exists(empty_flag_path)):
        if os.path.exists(info_path):
            os.remove(info_path)
        def _gen():
            path_list=[os.path.join(path_dir,fn) for fn in os.listdir(path_dir)]
            
            #imgs=[read_im(path) for path in path_list]
            for path in path_list:
                try:
                    if path[-4:] in ['flag','json']:
                        continue
                    print('Working on img:{}'.format(path))
                    img_src=cv2.imread(path)
                    w,h=img_src.shape[:2]
                    if len(img_src.shape)<3:
                        img_src=cv2.cvtColor(img_src,cv2.COLOR_GRAY2BGR)
                    batch_img_in = pre_processing([img_src])
        # =============================================================================
        #             print(path)
        #             cv2.imshow('',img)
        #             cv2.waitKey(1)
        # =============================================================================
                    
                except:
                    traceback.print_exc()
                    continue
                else:
                    yield batch_img_in,(w,h),img_src
                
        feature_list=[]
        gen_imgs=_gen()
        for batch_img_in,src_size,img_src in gen_imgs:
            nb_imgs+=1
            try:
                trt_outputs=_RETINAFACE.predict(batch_img_in)
                batch_results = post_processing( trt_outputs,src_size)

            except :
                traceback.print_exc()
                continue

            result=batch_results[0]
            rects,scores,landms=result
            rects,scores,landms=sub_by_scores(rects,scores,landms)
            have_one_face=len(rects)==1
            
            
            if have_one_face:
                landm=landms[0]#rect=rects[0]
            
                mat=get_transform_mat(landm.reshape(5,2),112)
                img_face=warp_img(mat,img_src,(112,112)).astype(np.float32)[None,:]/255
                emb=l2_norm(_ARCFACE.predict(img_face)[0])        
                feature_list.append(emb)
                
        if len(feature_list):
            feature_list=np.asarray(feature_list)
            mean_feature=np.mean(feature_list,axis=0)
            #print(path_dir+' nb feature:{}'.format(len(feature_list)))
            feature_list,mean_feature=_sub_feature(feature_list,mean_feature)
            _save_info(info_path,name_celeba,uid,sex_id,mean_feature)
            print("Computed facial feature vector from {} images from directory {}, got {} faces.\nActor's name is {}.".format(nb_imgs,path_dir,len(feature_list),name_celeba))
            flag_empty=False
            if os.path.exists(empty_flag_path):
                os.remove(empty_flag_path)
        else:
            flag_empty=True
            with open(empty_flag_path,'w') as fp:
                fp.write('e')
            mean_feature=None
    elif os.path.exists(empty_flag_path):
        flag_empty=True
        mean_feature=None
    else:
        name_celeba,uid,sex_id,mean_feature=_load_info(info_path)
        flag_empty=False
        if os.path.exists(empty_flag_path):
            os.remove(empty_flag_path)
    return name_celeba,uid,sex_id,mean_feature,flag_empty

def register_all(root_dir,out_path,use_par=False,over_write=False):
    if over_write:
        remove_old(root_dir)
    pathes_dir=[path_dir for path_dir in [os.path.join(root_dir,name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]
    
    results=[]
    
    print('Registing {} by serial processing'.format(root_dir))
    results=[worker_dir(path_dir,over_write) for path_dir in pathes_dir]
    
    info_dict={}
    
    for i,(name_celeba,uid,sex_id,mean_feature,flag_empty) in enumerate(results):
        if not flag_empty:
            info_dict[uid]={'index':int(i),'name':name_celeba,'sex':sex_id,'vector':mean_feature.tolist()}
    
    save_facial_infos(out_path,info_dict)
    

def register_cast(uid_sub,url_sub):
    from .spider import crawl_cast,get_cast,DEFAULT_IMGS_DIR
    from .database_ops import get_sub_facial_infos,dump_sub_infos,is_sub_dict_exist,get_sub_dict_path
    from utils.facial import load_facial_infos
    from utils.pathes import DEFAULT_DB_DIR,DEFAULT_UNI_DB_PATH
    if is_sub_dict_exist(DEFAULT_DB_DIR,uid_sub):
        path_sub_dict=get_sub_dict_path(DEFAULT_DB_DIR,uid_sub)
        sub_info_dict,uids_index,indecies,sorted_array=load_facial_infos(path_sub_dict)
    elif url_sub:
        name_list=get_cast(url_sub)
        crawl_cast(DEFAULT_IMGS_DIR,url_sub,use_par=True)
        register_all(DEFAULT_IMGS_DIR,DEFAULT_UNI_DB_PATH)
        sub_info_dict,uids_index,indecies,sorted_array=get_sub_facial_infos(DEFAULT_UNI_DB_PATH,name_list)
        dump_sub_infos(DEFAULT_DB_DIR,uid_sub,sub_info_dict)
    else:
        sub_info_dict,uids_index,indecies,sorted_array=load_facial_infos(DEFAULT_UNI_DB_PATH)
    
    return sub_info_dict,uids_index,indecies,sorted_array
