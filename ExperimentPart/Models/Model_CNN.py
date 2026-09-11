from os import path 
 
import torch 
 
import torch.nn as nn 
 
import sys 
 
sys.path.append("DataProcessPart") 
 
 
#---------create a class to load and adapt the data for the CNN model------------------# 
 
class Model_CNN(nn.Module): 
 
    def __init__(self, file_path, num_classes): 
 
        super().__init__() 
 
        #--get the dataset from the path 
 
        train_set, validation_set, test_set = self.load_data(file_path) 
 
        #--convert the BenchmarkData X to PyTorch tensors and stack them into a single tensor 
 
        train_X = self.adapt_data(train_set) 
 
        validation_X = self.adapt_data(validation_set) 
 
        test_X = self.adapt_data(test_set) 
 
        #-----normalize the data using the mean and standard deviation of the training set 
 
        mean = train_X.mean() 
 
        std = train_X.std() 
 
        train_X = (train_X - mean) / std 
 
        validation_X = (validation_X - mean) / std 
 
        test_X = (test_X - mean) / std 
 
        #--reshape the tensors to have a channel dimension of 1 
 
        train_X = train_X.unsqueeze(1) 
 
        validation_X = validation_X.unsqueeze(1) 
 
        test_X = test_X.unsqueeze(1) 
 
        #--create the label mapping 0-9 
 
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
         
        #--convert the labels to integer class indices 
 
        train_y = self.adapt_labels(train_set, label_to_index) 
 
        validation_y = self.adapt_labels( 
            validation_set, label_to_index 
        ) 
 
        test_y = self.adapt_labels( 
            test_set, label_to_index 
        ) 
 
        #--store the adapted data 
 
        self.train_X = train_X 
 
        self.validation_X = validation_X 
 
        self.test_X = test_X 
 
 
        self.train_y = train_y 
 
        self.validation_y = validation_y 
 
        self.test_y = test_y 
 
 
        self.label_to_index = label_to_index 
 
        #--create the CNN model 
 
        self.model = nn.Sequential( 
 
            nn.Conv1d(1, 32, kernel_size=7, padding=3), 
 
            nn.ReLU(), 
 
            nn.MaxPool1d(kernel_size=2), 
 
            nn.Conv1d(32, 64, kernel_size=5, padding=2), 
 
            nn.ReLU(), 
 
            nn.MaxPool1d(kernel_size=2), 
 
            nn.Conv1d(64, 128, kernel_size=3, padding=1), 
 
            nn.ReLU(), 
 
            nn.MaxPool1d(kernel_size=2), 
 
            nn.AdaptiveAvgPool1d(1), 
 
            nn.Flatten(), 
 
            nn.Linear(128, num_classes) 
 
        ) 
 
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
 
    @staticmethod 
    def adapt_data(data_set): 
 
        """Convert BenchmarkData X to PyTorch tensors.""" 
 
        X = [] 
 
        for data in data_set: 
 
            X.append(torch.tensor( 
                data.X, 
                dtype=torch.float32 
            )) 
 
        return torch.stack(X) 
 
    @staticmethod 
    def adapt_labels(data_set, label_to_index): 
 
        """Convert string labels to integer class indices.""" 
 
        y = [] 
 
        for data in data_set: 
 
            if data.y.endswith("-Normal"): 
                label = "Normal" 
            else: 
                label = "-".join(data.y.split("-")[1:]) 
 
            y.append(label_to_index[label]) 
 
        return torch.tensor(y, dtype=torch.long) 
 
    def forward(self, x): 
 
        return self.model(x)