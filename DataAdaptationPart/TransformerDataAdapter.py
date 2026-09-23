#--------------------------------------------------------
#--this file is used to adapt experiment data for transformer model.
#--------------------------------------------------------

import torch


class TransformerDataAdapter:

    def __init__(self, file_path):

        #--prepare and adapt the dataset
        self._prepare_data(file_path)


#-----------------------------------------------------------------------------------------------
    #--load data from specified path
    @staticmethod
    def load_data(file_path):
        """Load sampled data from a .pt file."""

        data = torch.load(
            file_path,
            weights_only=False,
        )

        train_set = data["training"]
        validation_set = data["validation"]
        test_set = data["test"]

        return train_set, validation_set, test_set


    #--transform loaded data into PyTorch tensor form.
    @staticmethod
    def adapt_data(data_set):
        """Convert X_data to PyTorch tensors."""

        X = []

        for data in data_set:

            X.append(
                torch.tensor(
                    data.X_data,
                    dtype=torch.float32
                )
            )

        return torch.stack(X)


    #--convert labels from string to integer form.
    @staticmethod
    def adapt_labels(data_set, label_to_index):
        """Convert string labels to integer class indices."""

        y = []

        for data in data_set:

            y.append(
                label_to_index[data.y]
            )

        return torch.tensor(
            y,
            dtype=torch.long
        )


#-----------------------------------------------------------------------------------------------
    #--adapt data for the transformer model.
    def _prepare_data(self, file_path):
        """Load and adapt the dataset for the Transformer model."""

        #--get the dataset from the path
        train_set, validation_set, test_set = self.load_data(
            file_path
        )

        #--convert X_data to PyTorch tensors
        train_X = self.adapt_data(train_set)
        validation_X = self.adapt_data(validation_set)
        test_X = self.adapt_data(test_set)

        #--normalize the data using the mean and standard deviation
        #--of the training set
        mean = train_X.mean()
        std = train_X.std()

        if std == 0:
            raise ValueError(
                "Standard deviation of the training data is zero. "
                "Normalization cannot be performed."
            )

        train_X = (train_X - mean) / std
        validation_X = (validation_X - mean) / std
        test_X = (test_X - mean) / std

        #--reshape the tensors for Transformer input
        #--shape: samples × sequence_length × feature_dimension
        train_X = train_X.unsqueeze(-1)
        validation_X = validation_X.unsqueeze(-1)
        test_X = test_X.unsqueeze(-1)

        #--create the label mapping using the training set only
        labels = sorted(
            {
                data.y
                for data in train_set
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