import torch
import torch.nn as nn
from os import path

from ExpermentPart.Models.Model_CNN import Model_CNN


#---------------------------------------------------------------
#--configurations
#---------------------------------------------------------------

FILE_PATH = "D:\Project\PHM_Data\SampledData\zy_10000_1024_512_ex028_10Lables"
FILE_NAME = "sampled_data_20260901164611.pt"
NUM_CLASSES = 10



model = Model_CNN(
    file_path=path.join(FILE_PATH, FILE_NAME),
    num_classes=NUM_CLASSES
)

criterion = nn.CrossEntropyLoss()

output = model(model.train_X)

loss = criterion(output, model.train_y)

print("-----------loss-----------")
print(loss)

loss.backward()

print("-----------backward-----------")
print("backward completed")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

optimizer.step()

print("-----------optimizer-----------")
print("optimizer step completed")