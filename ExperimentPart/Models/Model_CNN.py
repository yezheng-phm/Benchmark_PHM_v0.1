#-------------------------------------------------------------
#--this file is used to develop a model for CNN.
#--create standard CNN class.
#-------------------------------------------------------------

import torch
import torch.nn as nn

class Model_CNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        #--create the convolutional feature extractor
        self.feature_extractor = self._create_feature_extractor()

        #--create the global average pooling
        self.global_pool = self._create_global_pool()

        #--create the classifier
        self.classifier = self._create_classifier(num_classes)



#-------------------------------------------------------------
    #--create convolutional feature extractor method.
    def _create_feature_extractor(self):

        return nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU()
        )


    #--create global average pooling method.
    def _create_global_pool(self):

        return nn.AdaptiveAvgPool1d(1)


    #--make the domain of output is same with classification required.
    def _global_average_pooling(self, x):

        x = self.global_pool(x)

        return x.squeeze(-1)


    def _create_classifier(self, num_classes):

        return nn.Linear(
            128,
            num_classes
        )



    def forward(self, x):

        #--process the input using the convolutional feature extractor
        x = self.feature_extractor(x)

        #--aggregate the sequence representations
        x = self._global_average_pooling(x)

        x = self.classifier(x)

        return x