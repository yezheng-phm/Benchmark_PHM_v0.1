import torch
from os import path

import sys
sys.path.append("DataProcessPart")


FILE_PATH = "D:\Project\PHM_Data\SampledData\zy_10000_1024_512_ex028_10Lables"
FILE_NAME = "sampled_data_20260901164611.pt"

data = torch.load(
    path.join(FILE_PATH, FILE_NAME),
    weights_only=False
)

train_set = data["train"]
validation_set = data["validation"]
test_set = data["test"]

labels = sorted({
    item.y
    for item in train_set + validation_set + test_set
})

print("-----------original labels-----------")

for label in labels:
    print(label)

print("-----------number of labels-----------")
print(len(labels))