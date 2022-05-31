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
from Unet import UnetResNet

# from tvm_funcs import get_tvm_model, tune, time_it

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        def dict_to_tuple(out):
            output = np.squeeze(out)
            output_predictions = output.argmax(0)
            return output_predictions

        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])

        inp = torch.unsqueeze(inp, 0)
        # from bgr to rgb
        # inp = torch.flip(inp, [3])
        inp = inp.type(torch.float32) / 255.0
        inp[0, :, :, :] = (inp[0, :, :, :] - MEAN) / STD
        inp = inp.permute(0, 3, 1, 2)

        original_size = inp.size()
        input_size = (1, 3, 640, 1280)
        # inp.resize_(input_size)

        out = self.model(inp)
        labels = dict_to_tuple(out)
        # labels.resize_(original_size[-2:])
        return labels

def get_model():
    model = UnetResNet(encoder_name="resnext50",
                       num_classes=20,
                       input_channels=3,
                       num_filters=32,
                       Dropout=0.2,
                       res_blocks_dec=False)
    model.eval()

    model_path = "/home/sean/workspace/ros2_tvm/model/UNET_8x_downsize_all_classes/best_model.pth"
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    model = TraceWrapper(model)
    return model

def plot_segmentation(input_img, seg_image):
    num_classes = 20

    fig, axs = plt.subplots(1, 2, figsize=(16, 16))

    images = []

    axs[0].set_title("Image")
    axs[1].set_title("Prediction")

    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    images.append(axs[0].imshow(input_img.astype(int)))
    images.append(axs[1].imshow(seg_image, cmap=plt.get_cmap('nipy_spectral'), vmin=0, vmax=num_classes))

    seg_classes = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "void",
    ]

    cbar = fig.colorbar(images[1], ax=axs, orientation='horizontal', ticks=[x for x in range(num_classes)], fraction=.1)
    cbar.ax.set_xticklabels(list(seg_classes), rotation=55)

    plt.show()
    plt.pause(0)

if __name__ == "__main__":
    image = cv2.imread("/home/sean/workspace/ros2_tvm/data/2011_09_26-0056-0000000081-003157.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 256))
    input_img = torch.from_numpy(image)

    model = get_model()
    output = model(input_img)

    plot_segmentation(image, output)

    print(f"Converting the model (post-training)...")
    start_time = time.time()
    quantized = torch.quantization.convert(model)
    print(f"Quantization done in {str(time.time() - start_time)} seconds.")
    torch.save(quantized.state_dict(), "/home/sean/workspace/ros2_tvm/model/quantized.pth")

    # model = torch.jit.trace(model, image)

    # print("PyTorch (unquantized) timings:")
    # print(time_it(lambda: model(image)))
    #
    # print("PyTorch (quantized) timings:")
    # print(time_it(lambda: quantized_mobilenet(image)))

    # # tvm part
    # mod, params, module, lib = get_tvm_model(quantized, image)
    # # tvm_optimized_module = tune(mod, params, image)
    #
    # # save and load the code and lib file.
    # dir = "/home/sean/workspace/ros2_tvm/model/"
    # path_lib = os.path.join(dir, "segmentation_lib.so")
    # lib.export_library(path_lib)

    # print("TVM (Relay) timings:")
    # print(time_it(lambda: module.run()))
    # print("TVM (Tuned) timings:")
    # print(time_it(lambda: tvm_optimized_module.run()))