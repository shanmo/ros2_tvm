import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import pandas as pd

# module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
# detector = hub.load(module_handle).signatures['default']

# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1").signatures['serving_default']
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

def preprocess(image):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image -= MEAN
    image /= STD
    # tf is channel last
    # ref https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/
    # image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    return image

img = cv2.imread("/home/sean/workspace/ros2_tvm/data/2011_09_26-0056-0000000081-003157.png")
img = cv2.resize(img, (848, 480))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
converted_img = np.expand_dims(img, 0)
# converted_img = preprocess(img)
rgb_tensor  = tf.convert_to_tensor(converted_img, tf.uint8)
# result = detector(rgb_tensor)
# boxes, scores, classes, num_detections = result["detection_boxes"], result["detection_scores"], result["detection_class_entities"], len(result["detection_scores"])
boxes, scores, classes, num_detections = detector(rgb_tensor)

labels = pd.read_csv('/home/sean/workspace/ros2_tvm/scripts/labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Processing outputs
pred_labels = classes.numpy().astype('int')[0]
pred_labels = [labels[i] for i in pred_labels]
pred_boxes = boxes.numpy()[0].astype('int')
pred_scores = scores.numpy()[0]

# Processing outputs
# pred_labels = classes.numpy().astype('int')[0]
# pred_labels = [labels[i] for i in pred_labels]
# pred_boxes = boxes.numpy().astype('int')
# pred_scores = scores.numpy()

# Putting the boxes and labels on the image
img_boxes = img.copy()
for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < 0.5:
        continue

    score_txt = f'{100 * round(score)}%'
    img_boxes = cv2.rectangle(img_boxes, (xmin, ymax), (xmax, ymin),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)

#Display the resulting frame
# cv2.imshow('detection', img_boxes)
# cv2.waitKey(0)

graph_def = detector.signatures['serving_default'].graph.as_graph_def()

# tvm, relay
import tvm
from tvm import te
from tvm import relay

# error
# TVMError: In function ir.TensorType(0: Array<PrimExpr>, 1: DataType) -> relay.TensorType: error while converting argument 1: [14:27:26] /home/sean/mylibs/tvm/include/tvm/runtime/data_type.h:374: unknown type resource
target = tvm.target.Target("llvm", host="llvm")
# dev = tvm.cpu(0)
shape_dict = {"input0": converted_img.shape}
mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

# save and load the code and lib file.
dir = "/home/sean/workspace/ros2_tvm/model/"
path_lib = os.path.join(dir, "detection_lib.so")
lib.export_library(path_lib)


