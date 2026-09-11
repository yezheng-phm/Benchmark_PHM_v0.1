#-------------------------------------------------------------
#--this file is used to develop a model for transformer.
#--this file have two core fuctions:
#--1.data adaptation.
#--2.create standard transformer class.
#-------------------------------------------------------------

import torch
import torch.nn as nn

class Model_Transformer(nn.Module):

    def __init__(self, file_path, num_classes):

        super().__init__()

        #--prepare and adapt the dataset
        self._prepare_data(file_path)

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

        self.classifier = self._create_classifier(num_classes)


    #--prepare the data for transformer module.
    def _prepare_data(self, file_path):
        """Load and adapt the dataset for the Transformer model."""

        #--get the dataset from the path
        train_set, validation_set, test_set = self.load_data(
            file_path
        )

        #--convert BenchmarkData X to PyTorch tensors
        train_X = self.adapt_data(train_set)
        validation_X = self.adapt_data(validation_set)
        test_X = self.adapt_data(test_set)

        #--normalize the data using the mean and standard deviation
        #--of the training set
        mean = train_X.mean()
        std = train_X.std()

        train_X = (train_X - mean) / std
        validation_X = (validation_X - mean) / std
        test_X = (test_X - mean) / std

        #--reshape the tensors for Transformer input
        #--shape: samples × sequence_length × feature_dimension
        train_X = train_X.unsqueeze(-1)
        validation_X = validation_X.unsqueeze(-1)
        test_X = test_X.unsqueeze(-1)

        #--create the label mapping
        labels = sorted(
            {
                "Normal"
                if data.y.endswith("-Normal")
                else "-".join(data.y.split("-")[1:])
                for data in train_set + validation_set + test_set
            }
        )

        label_to_index = {
            label: index
            for index, label in enumerate(labels)
        }

        #--convert string labels to integer class indices
        train_y = self.adapt_labels(
            train_set,
            label_to_index
        )

        validation_y = self.adapt_labels(
            validation_set,
            label_to_index
        )

        test_y = self.adapt_labels(
            test_set,
            label_to_index
        )

        #--store the adapted data
        self.train_X = train_X
        self.validation_X = validation_X
        self.test_X = test_X

        self.train_y = train_y
        self.validation_y = validation_y
        self.test_y = test_y

        self.label_to_index = label_to_index


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


    #--data load method for loading experiment data.
    @staticmethod
    def load_data(file_path):
        """Load sampled data from a .pt file."""

        data = torch.load(
            file_path,
            weights_only=False,
        )

        train_set = data["train"]
        validation_set = data["validation"]
        test_set = data["test"]

        return train_set, validation_set, test_set


    #--transfor dataset to Pytorch Tensor.
    @staticmethod
    def adapt_data(data_set):
        """Convert BenchmarkData X to PyTorch tensors."""

        X = []

        for data in data_set:

            X.append(
                torch.tensor(
                    data.X,
                    dtype=torch.float32
                )
            )

        return torch.stack(X)


    #--transfor labels to Pytorch Tensor.
    @staticmethod
    def adapt_labels(data_set, label_to_index):
        """Convert string labels to integer class indices."""

        y = []

        for data in data_set:

            if data.y.endswith("-Normal"):
                label = "Normal"
            else:
                label = "-".join(
                    data.y.split("-")[1:]
                )

            y.append(
                label_to_index[label]
            )

        return torch.tensor(
            y,
            dtype=torch.long
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



