import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
import mxnet as mx

import numpy as np
import cv2, os

def preprocess(image):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image -= MEAN
    image /= STD
    ratio = 512.0 / 480
    image = cv2.resize(image, (round(848.0 * ratio), round(480.0 * ratio)))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    return image

img = cv2.imread("/home/sean/workspace/ros2_tvm/data/2011_09_26-0056-0000000081-003157.png")
img = cv2.resize(img, (848, 480))
input_img = preprocess(img)
dshape = input_img.shape
x = mx.nd.array(input_img)

model = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
# model = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)

def build(target):
    mod, params = relay.frontend.from_mxnet(model, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    return lib

def run(lib, dev):
    # Build TVM runtime
    m = graph_executor.GraphModule(lib["default"](dev))
    tvm_input = tvm.nd.array(x.asnumpy(), device=dev)
    m.set_input("data", tvm_input)
    # execute
    m.run()
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs

# for target in ["llvm", "cuda"]:
for target in ["llvm"]:
    dev = tvm.device(target, 0)
    if dev.exist:
        lib = build(target)
        class_IDs, scores, bounding_boxs = run(lib, dev)

def postprocess_bbox(bboxes):
    ratio = 512 / 480
    bboxes[:, (0, 2)] /= ratio
    bboxes[:, (1, 3)] /= ratio
    return bboxes

# save and load the code and lib file.
dir = "/home/sean/workspace/ros2_tvm/model/"
path_lib = os.path.join(dir, "detection_lib.so")
lib.export_library(path_lib)

bbox = postprocess_bbox(bounding_boxs.numpy()[0])
print(f"bbox shape {bbox.shape}")

ax = utils.viz.plot_bbox(
    img,
    bbox,
    scores.numpy()[0],
    class_IDs.numpy()[0],
    class_names=model.classes,
)
plt.show()


