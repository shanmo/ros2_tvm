# Model code adpated from:
# https://github.com/spellml/mobilenet-cifar10/blob/master/servers/eval_quantized_t4.py
import math
import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from tvm_funcs import get_tvm_model, tune, time_it

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        out = self.model(input)
        return out[0]["boxes"], out[0]["scores"], out[0]["labels"]

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model = TraceWrapper(model)
    return model

def preprocess(image):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image -= MEAN
    image /= STD
    image = cv2.resize(image, (512, 256))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    return image

if __name__ == "__main__":
    image = cv2.imread("/home/sean/workspace/ros2_tvm/data/2011_09_26-0056-0000000081-003157.png")
    image = preprocess(image)
    input_img = torch.from_numpy(image)

    model = get_model()
    output = model(input_img)
    # print(output)

    print(f"Converting the model (post-training)...")
    start_time = time.time()
    quantized = torch.quantization.convert(model)
    print(f"Quantization done in {str(time.time() - start_time)} seconds.")
    torch.save(quantized.state_dict(), "/home/sean/workspace/ros2_tvm/model/quantized.pth")

    # print("PyTorch (unquantized) timings:")
    # print(time_it(lambda: model(image)))
    #
    # print("PyTorch (quantized) timings:")
    # print(time_it(lambda: quantized(image)))

    # tvm part
    mod, params, module, lib = get_tvm_model(quantized, input_img)
    # tvm_optimized_module = tune(mod, params, image)

    # save and load the code and lib file.
    dir = "/home/sean/workspace/ros2_tvm/model/"
    path_lib = os.path.join(dir, "detection_lib.so")
    lib.export_library(path_lib)

