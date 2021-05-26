#!/usr/bin/python

import json
idx2label = []
cls2label = {}

# The imagenet_class_index.json file is from Keras for imagenet human name mapping
# https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json

with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}


with open("training_categories.txt", "r") as f:
    lines = f.readlines()
    for cls in lines:
        cls = cls.strip()
        idx = cls2label[cls]
        print("{}, {}".format(cls, idx))

