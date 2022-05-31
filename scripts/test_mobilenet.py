# Model code adpated from:
# https://github.com/spellml/mobilenet-cifar10/blob/master/servers/eval_quantized_t4.py
import math
import os
import time
import numpy as np
import cv2

import torch
from torch import optim
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from tvm_funcs import get_tvm_model, tune, time_it

def preprocess(loaded):
    # CV loads in BGR, and rcnn expects rgb
    loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    img_data = loaded.transpose(2, 0, 1).astype(np.float32)/255.0
    norm_img_data = np.expand_dims(img_data, axis=0)
    return norm_img_data

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        def dict_to_tuple(out_dict):
            return out_dict["out"]

        out = self.model(inp)
        # print(out.keys())
        return dict_to_tuple(out)

def get_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model = TraceWrapper(model)
    model.eval()
    return model

if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     dev = "cuda:0"
    # else:
    #     dev = "cpu"
    dev = "cpu"

    image = cv2.imread("/home/sean/workspace/ros2_tvm/data/2011_09_26-0056-0000000081-003157.png")
    image = preprocess(image)
    image = torch.from_numpy(image).to(dev)

    mobilenet = get_model()
    mobilenet.to(dev)

    print(f"Converting the model (post-training)...")
    start_time = time.time()
    quantized_mobilenet = torch.quantization.convert(mobilenet)
    print(f"Quantization done in {str(time.time() - start_time)} seconds.")
    torch.save(quantized_mobilenet.state_dict(), "/home/sean/workspace/ros2_tvm/model/quantized_model.pth")

    # print("PyTorch (unquantized) timings:")
    # print(time_it(lambda: mobilenet(image)))
    #
    # print("PyTorch (quantized) timings:")
    # print(time_it(lambda: quantized_mobilenet(image)))

    # tvm part
    mod, params, module, lib = get_tvm_model(quantized_mobilenet, image)
    # tvm_optimized_module = tune(mod, params, image)

    # save and load the code and lib file.
    dir = "/home/sean/workspace/ros2_tvm/model/"
    path_lib = os.path.join(dir, "segmentation_lib.so")
    lib.export_library(path_lib)

    # print("TVM (Relay) timings:")
    # print(time_it(lambda: module.run()))
    # print("TVM (Tuned) timings:")
    # print(time_it(lambda: tvm_optimized_module.run()))