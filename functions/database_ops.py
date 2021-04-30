# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:12:09 2018

@author: quantumliu
"""

import os
import json

import numpy as np

from .spider import get_cast
from utils.facial import load_facial_infos

def get_sub_facial_infos(path,name_list):
    with open(path,'r') as fp:
        info_dict=json.load(fp)
    #sorted_uids=sorted(info_dict.keys())
    name2uid={v['name']:k for k,v in info_dict.items()}
    #print(name2uid)
    sub_info_dict={}
    for name in name_list:
        #print(name)
        k_uid=name2uid.get(name,False)
        if k_uid:
            sub_info_dict[k_uid]=info_dict[k_uid]
    
    uids_index={v['index']:k for k,v in sub_info_dict.items()}
    
    indecies=sorted(list(uids_index.keys()))
    
    sorted_array=np.asarray([sub_info_dict[uids_index[i]]['vector'] for i in indecies])
    #print(sub_info_dict)
    return sub_info_dict,uids_index,indecies,sorted_array

def get_sub_dict_path(root_dir,id_sub):
    target_path=os.path.join(root_dir,id_sub+'.json')
    print('target_path:',target_path)
    return target_path

def get_sub_dict_path_3d(root_dir,id_sub):
    target_path=os.path.join(root_dir,id_sub+'_3d.json')
    print('target_path:',target_path)
    return target_path

def is_sub_dict_exist(root_dir,id_sub):
    target_path=get_sub_dict_path(root_dir,id_sub)
    return os.path.exists(target_path)

def dump_sub_infos(root_dir,id_sub,info_sub,overwrite=False):
    target_path=get_sub_dict_path(root_dir,id_sub)
    if (not is_sub_dict_exist(root_dir,id_sub)) or overwrite:
        with open(target_path,'w') as fp:
            json.dump(info_sub,fp)
    
