import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd

prefix, epoch = "/home/thongpb/works/my_insightface/my_insightface/models/model-y1-test2/model", 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
image_size = (112, 112)
opt_level = 0

shape_dict = {'data': (1, 3, 112, 112)}
target = tvm.target.create("llvm -mcpu=haswell")
# "target" means your target platform you want to compile.

#target = tvm.target.create("llvm -mcpu=broadwell")
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
with nnvm.compiler.build_config(opt_level=opt_level):
   graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
lib.export_library("./deploy_lib.so")
print('lib export succeefully')
with open("./deploy_graph.json", "w") as fo:
   fo.write(graph.json())
with open("./deploy_param.params", "wb") as fo:
   fo.write(nnvm.compiler.save_param_dict(params))
