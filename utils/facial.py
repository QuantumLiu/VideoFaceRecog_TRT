import numpy as np
import os
import json

from models.arcface.utils import l2_norm
def _parse_info(path_dir):
    name_dir=os.path.split(path_dir)[-1]
    
    name_celeba,sex_char=name_dir.split('_')
    sex_id={'m':0,'f':1}[sex_char]
    return name_celeba,sex_id

def _save_info(path,name_celeba,uid,sex_id,mean_feature):
    info_tuple=(name_celeba,uid,sex_id,mean_feature.tolist())
    save_facial_infos(path,info_tuple)
    
def _load_info(path):
    with open(path,'r') as fp:
        info_tuple=json.load(fp)
    name_celeba,uid,sex_id,mean_feature=info_tuple
    mean_feature=np.asarray(mean_feature)
    return name_celeba,uid,sex_id,mean_feature
def face_distance(known_face_encoding,face_encoding_to_check):
    fl=np.asarray(known_face_encoding)
    return np.dot(fl,face_encoding_to_check)

def face_identify(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    distance=face_distance(known_face_encoding, face_encoding_to_check)
    
    argmax=np.argmax(distance)
    d_min=distance[argmax]

    if distance[argmax]<tolerance:
        index=-1
        is_known=False
    else:
        index=argmax
        is_known=True
    return is_known,index,d_min

def _sub_feature(feature_list,mean_feature,rate=0.9):
    nb_feature=int(rate*len(feature_list))
    if nb_feature:
        dists=face_distance(feature_list,mean_feature)
        
        sub_feature_list= feature_list[np.argsort(dists)[:nb_feature]]
        mean_feature=l2_norm(np.mean(sub_feature_list,axis=0))
        return sub_feature_list,mean_feature
    else:
        return feature_list.copy(),feature_list[0].copy()

def save_facial_infos(path,info_dict):
    with open(path,'w') as fp:
        json.dump(info_dict,fp)
    
def load_facial_infos(path):
    with open(path,'r') as fp:
        info_dict=json.load(fp)
    #sorted_uids=sorted(info_dict.keys())
    uids_index={v['index']:k for k,v in info_dict.items()}
    indecies=sorted(list(uids_index.keys()))
    
    sorted_array=np.asarray([info_dict[uids_index[i]]['vector'] for i in indecies])
    return info_dict,uids_index,indecies,sorted_array

def query_id(array_index,info_dict,uids_index,indecies):
    real_index=indecies[array_index]
    uid=uids_index[real_index]
    name=info_dict[uid]['name']
    sex=info_dict[uid]['sex']
    return uid,real_index,name,sex

def query_id_batch(batch_result,info_dict,uids_index,indecies):
    out_batch_result=[]
    for sample_index,have_face,frame_result in batch_result:
        out_frame_result=[]
        for is_known,array_index,distance,rect_tuple_ori in frame_result:
            real_index=indecies[array_index]
            uid=uids_index[real_index]
            name=info_dict[uid]['name']
            sex=info_dict[uid]['sex']
            out_frame_result.append((is_known,(uid,name,sex,distance),rect_tuple_ori))
        out_batch_result.append((sample_index,have_face,out_frame_result))
    return out_batch_result

def query_array_by_name(info_dict,name):
    name2array={v['name']:v['vector'] for v in info_dict.values()}
    if not name in name2array.keys():
        return False,None
    else:
        known_array=np.asarray(name2array[name])
        return True,known_array

def _default_int(o):
    if isinstance(o, (np.int64,np.int32)): 
        return int(o)
    if isinstance(o,(np.float16,np.float32,np.float64)):
        return float(o)
    raise TypeError

def remove_old(root_dir):
    pathes_dir=[path_dir for path_dir in [os.path.join(root_dir,name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]
    for path_dir in pathes_dir:
        info_path=os.path.join(path_dir,'info.json')
        if os.path.exists(info_path):
            os.remove(info_path)
