#-------------------------------------------------------------
#--this file is used to develop a model for transformer.
#--create standard transformer class.
#-------------------------------------------------------------

import torch
import torch.nn as nn

class Model_Transformer(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        #--create the input projection
        self.input_projection = self._create_input_projection()

        #--create the positional encoding
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(
                sequence_length=1024,
                d_model=128
            )
        )

        #--create the transformer encoder
        self.transformer_encoder = self._create_transformer_encoder()

        #--create the classifier
        self.classifier = self._create_classifier(num_classes)



#-------------------------------------------------------------
    #--create input projection method.
    def _create_input_projection(self):

        return nn.Linear(
            1,
            128
        )


    #--create transformer encoder method.
    def _create_transformer_encoder(self):

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )


    #--make the domain of output is same with classification required.
    def _global_average_pooling(self, x):

        return x.mean(dim=1)


    def _create_classifier(self, num_classes):

        return nn.Linear(
            128,
            num_classes
        )


    #--create positional tensor.
    @staticmethod
    def _create_positional_encoding(sequence_length, d_model):

        position = torch.arange(
            sequence_length,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        positional_encoding = torch.zeros(
            sequence_length,
            d_model
        )

        positional_encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        positional_encoding[:, 1::2] = torch.cos(
            position * div_term
        )

        return positional_encoding.unsqueeze(0)



    def forward(self, x):

        #--project the input feature dimension
        x = self.input_projection(x)

        #--add positional information
        x = x + self.positional_encoding

        #--process the sequence using the transformer encoder
        x = self.transformer_encoder(x)

        #--aggregate the sequence representations
        x = self._global_average_pooling(x)

        x = self.classifier(x)

        return x



