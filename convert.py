import os
path_bin="/home/ubuntu/Downloads/TensorRT-7.1.3.4/bin/trtexec"
temp_cmd='''{} --onnx="{}" --explicitBatch --saveEngine="{}" --workspace=8192 --fp16 '''
path_dir_onnx="onnx_models"
fns_onnx=os.listdir(path_dir_onnx)
for fn in fns_onnx:
    path_onnx=os.path.join(path_dir_onnx,fn)
    path_engine=os.path.join('trt_engines',os.path.splitext(fn)[0]+'.engine')
    cmd=temp_cmd.format(path_bin,path_onnx,path_engine)
    os.system(cmd)

