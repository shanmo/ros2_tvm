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

supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc" "ssd_512_vgg16_atrous_coco",
]

model_name = supported_model[0]
dshape = (1, 3, 1242, 375)

def preprocess(image):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image -= MEAN
    image /= STD
    # image = cv2.resize(image, (512, 512))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    return image

img = cv2.imread("/home/sean/workspace/ros2_tvm/data/2011_09_26-0056-0000000081-003157.png")
input_img = preprocess(img)
x = mx.nd.array(input_img)

# im_fname = download_testdata(
#     "https://github.com/dmlc/web-data/blob/main/" + "gluoncv/detection/street_small.jpg?raw=true",
#     "street_small.jpg",
#     module="data",
#     )
# x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

block = model_zoo.get_model(model_name, pretrained=True)


def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
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

# save and load the code and lib file.
dir = "/home/sean/workspace/ros2_tvm/model/"
path_lib = os.path.join(dir, "detection_lib.so")
lib.export_library(path_lib)

ax = utils.viz.plot_bbox(
    img,
    bounding_boxs.numpy()[0],
    scores.numpy()[0],
    class_IDs.numpy()[0],
    class_names=block.classes,
)
plt.show()


