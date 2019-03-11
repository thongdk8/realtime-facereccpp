import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd

ctx = tvm.cpu()
data_shape = (3,112,112)
# load the module back.
loaded_json = open("./deploy_graph.json").read()
loaded_lib = tvm.module.load("./deploy_lib.so")
loaded_params = bytearray(open("./deploy_param.params", "rb").read())

input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

# Tiny benchmark test.
import time
for i in range(100):
   t0 = time.time()
   module.run(data=input_data)
   print(1/(time.time() - t0))