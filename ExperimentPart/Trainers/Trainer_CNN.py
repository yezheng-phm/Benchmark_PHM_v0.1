#---------------------------------------------------------
#--this file is a trainer for the Model_CNN, which is used to train the CNN model on the given dataset.
#---------------------------------------------------------

import os
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ExperimentPart.ExperimentResultContainer import (
    ExperimentResultContainer,
    EpochResult
)

class Trainer_CNN:
    """
    A trainer class for the Model_CNN model.
    """    
    def __init__(
        self,
        model,
        batch_size,
        epochs,
        runs_num,
        learning_rate,
        optimizer,
        criterion,
        base_seed,
        experiment_name,
        experiment_id,
        experiment_data_path,
        features_save_layers
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.runs_num = runs_num
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_instance = None
        self.criterion = criterion
        self.criterion_instance = None
        self.base_seed = base_seed
        self.features_save_layers = features_save_layers

#--create a layers names registry mapping table with the input model.
        self.feature_registry = self._get_feature_registry()

#--verify the input layers name that user wants to extract features is legal or not.
        self.features_save_layers = (
            self._validate_input_layers_name_protocol(
                self.features_save_layers
            )
        )

#--validate the experiment_id to ensure it is a digit and format it accordingly.
        if not str(experiment_id).isdigit():
            raise ValueError("Experiment_ID is not a digit and must be a digit!")
        
#--format the experiment_id to have a prefix "Ex" and be zero-padded to three digits.
        experiment_id = int(experiment_id)
        experiment_id = f"Ex{experiment_id:03d}"

#--validate the experiment data path to ensure it is not empty.
        if not experiment_data_path:
            raise ValueError("ExperimentDataPath cannot be empty. Please provide the experiment data save path!")
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.experiment_data_path = experiment_data_path

#--create a method to build the optimizer according to the configuration.
    def _build_optimizer(self):
        """Build the optimizer according to the configuration."""

        if self.optimizer == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )

        raise ValueError(
            f"Unsupported optimizer: {self.optimizer}"
        )

#--create a method to build the loss function according to the configuration.
    def _build_criterion(self):
        """Build the loss function according to the configuration."""

        if self.criterion == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()

        raise ValueError(
            f"Unsupported criterion: {self.criterion}"
        )

#--create a method to train the model for one epoch.
    def _train_one_epoch(self, train_loader):
        """Train the model for one epoch."""

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        #--start the timer for the epoch training time.
        start_time = time.time()

        for X, y in train_loader:

            self.optimizer_instance.zero_grad()

            output = self.model(X)

            loss = self.criterion_instance(output, y)

            loss.backward()

            self.optimizer_instance.step()

            total_loss += loss.item() * X.size(0)

            predicted = output.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        #--get the training time for one epoch.
        epoch_train_time = time.time() - start_time

        return epoch_loss, epoch_accuracy, epoch_train_time



#--used for verifying the model on the validation dataset for one epoch.
    def _validate_one_epoch(self, validation_loader):
        """Validate the model for one epoch."""

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for X, y in validation_loader:

                output = self.model(X)

                loss = self.criterion_instance(output, y)

                total_loss += loss.item() * X.size(0)

                predicted = output.argmax(dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    
#--create DataLoaders for the training, validation, and test datasets.
    def _build_dataloader(self):
        """Build DataLoaders for training, validation, and test datasets."""

        train_dataset = TensorDataset(
            self.model.train_X,
            self.model.train_y
        )

        validation_dataset = TensorDataset(
            self.model.validation_X,
            self.model.validation_y
        )

        test_dataset = TensorDataset(
            self.model.test_X,
            self.model.test_y
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, validation_loader, test_loader



#--core train logic for training the model for one run and selecting the best epoch based on validation performance.
    #--core train logic for training the model for one run and selecting the best epoch based on validation performance.
    def _train_one_run(
        self,
        train_loader,
        validation_loader,
        run_id
    ):
        """Train the model for one run and select the best epoch based on validation performance."""

        epoch_info_list = []

        selected_epoch_num = None
        best_validation_accuracy = None
        best_validation_loss = None
        best_epoch_train_time = None
        best_model_state = None


        #--build the optimizer for the current run.
        self.optimizer_instance = self._build_optimizer()

        #--build the loss function for the current run.
        self.criterion_instance = self._build_criterion()


        #--create the progress bar for the current run.
        epoch_progress = tqdm(
            range(1, self.epochs + 1),
            desc=f"Run {run_id}/{self.runs_num}",
            leave=True
        )


        for epoch in epoch_progress:

            epoch_train_loss, epoch_train_accuracy, epoch_train_time = (
                self._train_one_epoch(train_loader)
            )

            epoch_validation_loss, epoch_validation_accuracy = (
                self._validate_one_epoch(validation_loader)
            )

            epoch_result = EpochResult(
                epoch_num=epoch,
                epoch_train_loss=epoch_train_loss,
                epoch_train_accuracy=epoch_train_accuracy,
                epoch_train_time=epoch_train_time,
                epoch_validation_loss=epoch_validation_loss,
                epoch_validation_accuracy=epoch_validation_accuracy
            )

            epoch_info_list.append(epoch_result)


            #--update the training progress.
            epoch_progress.set_postfix(
                train_loss=f"{epoch_train_loss:.3f}",
                val_acc=f"{epoch_validation_accuracy:.3f}"
            )


            if selected_epoch_num is None:

                selected_epoch_num = epoch
                best_validation_accuracy = epoch_validation_accuracy
                best_validation_loss = epoch_validation_loss
                best_epoch_train_time = epoch_train_time
                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

            elif epoch_validation_accuracy > best_validation_accuracy:

                selected_epoch_num = epoch
                best_validation_accuracy = epoch_validation_accuracy
                best_validation_loss = epoch_validation_loss
                best_epoch_train_time = epoch_train_time
                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

            elif (
                epoch_validation_accuracy == best_validation_accuracy
                and epoch_validation_loss < best_validation_loss
            ):

                selected_epoch_num = epoch
                best_validation_accuracy = epoch_validation_accuracy
                best_validation_loss = epoch_validation_loss
                best_epoch_train_time = epoch_train_time
                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

            elif (
                epoch_validation_accuracy == best_validation_accuracy
                and epoch_validation_loss == best_validation_loss
                and epoch_train_time < best_epoch_train_time
            ):

                selected_epoch_num = epoch
                best_epoch_train_time = epoch_train_time
                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )


        self.model.load_state_dict(
            best_model_state
        )

        return epoch_info_list, selected_epoch_num


    #--create a method to test the selected model on the test dataset.
    def _test_one_run(self, test_loader):
        """Test the selected model on the test dataset and return the test information."""

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        y_true = []
        y_pred = []

        with torch.no_grad():

            for X, y in test_loader:

                output = self.model(X)

                loss = self.criterion_instance(output, y)

                total_loss += loss.item() * X.size(0)

                predicted = output.argmax(dim=1)

                correct += (predicted == y).sum().item()
                total += y.size(0)

                y_true.extend(y.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())

        test_loss = total_loss / total
        test_accuracy = correct / total

        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "y_true": y_true,
            "y_pred": y_pred,
            "label_to_index": self.model.label_to_index
        }    


#--create a method to save the selected model of the current run and return its information.
    def _get_model_info(self, run_id, selected_epoch_num):
        """Save the selected model of the current run and return its information."""

        experiment_name = f"{self.experiment_name}_{self.experiment_id}"

        experiment_dir = os.path.join(
            self.experiment_data_path,
            experiment_name
        )

        run_dir = os.path.join(
            experiment_dir,
            f"Run_{run_id:03d}"
        )

        model_dir = os.path.join(
            run_dir,
            "model"
        )

        os.makedirs(model_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            model_dir,
            "selected_model.pt"
        )

        torch.save(
            self.model.state_dict(),
            checkpoint_path
        )

        relative_checkpoint_path = os.path.relpath(
            checkpoint_path,
            self.experiment_data_path
        )

        return {
            "checkpoint_path": relative_checkpoint_path,
            "selected_model_epoch_num": selected_epoch_num
        }


#--create the feature registry for the current CNN model.
    def _get_feature_registry(self):
        """Create the feature registry according to the actual CNN model layers."""

        feature_registry = {}

        conv_id = 1000
        pool_id = 2000

        conv_num = 0
        pool_num = 0

        for layer in self.model.model:

            if isinstance(layer, nn.Conv1d):
                feature_registry[conv_id] = {
                    "name": f"conv{conv_num + 1}",
                    "layer": layer
                }

                conv_id += 1
                conv_num += 1

            elif isinstance(layer, nn.MaxPool1d):
                feature_registry[pool_id] = {
                    "name": f"pool{pool_num + 1}",
                    "layer": layer
                }

                pool_id += 1
                pool_num += 1

            elif isinstance(layer, nn.AdaptiveAvgPool1d):
                feature_registry[9000] = {
                    "name": "globalpool",
                    "layer": layer
                }

        feature_registry[10000] = {
            "name": "all",
            "layer": None
        }

        return feature_registry



#--validate the input feature layer names according to the CNN layer name protocol.
    def _validate_input_layers_name_protocol(self, input_layers):
        """Validate and standardize the input feature layer names."""

        if not isinstance(input_layers, list):
            raise TypeError(
                "Features_Save_Layers must be a list."
            )

        standardized_layers = []

        valid_layer_names = {
            info["name"]
            for info in self.feature_registry.values()
            if info["name"] != "all"
        }

        for layer_name in input_layers:

            if not isinstance(layer_name, str):
                raise TypeError(
                    "Each feature layer name must be a string."
                )

            layer_name = layer_name.lower()

            if layer_name == "all":
                standardized_layers.append(layer_name)
                continue

            if layer_name not in valid_layer_names:
                raise ValueError(
                    f"Invalid feature layer name: '{layer_name}'. "
                    f"Available feature layers: "
                    f"{sorted(valid_layer_names)}."
                )

            standardized_layers.append(layer_name)

        return list(dict.fromkeys(standardized_layers))




#--get the features from the model layers that you wanted. 
#--feature shape is represented as samples_num*out_channel*feature_points

    def _get_features_info(self, test_loader, run_id, selected_epoch_num):
        """Extract and save the selected model features from the test dataset."""

        self.model.eval()

        features = {}

        if "all" in self.features_save_layers:

            selected_layers = [
                info["name"]
                for layer_id, info in self.feature_registry.items()
                if info["name"] != "all"
            ]

        else:

            selected_layers = self.features_save_layers

        for layer_name in selected_layers:

            feature_id = None

            for layer_id, info in self.feature_registry.items():

                if info["name"] == layer_name:

                    feature_id = layer_id
                    break

            features[layer_name] = {
                "feature_id": feature_id,
                "data": []
            }

        hooks = []
        feature_outputs = {}

        for layer_name in selected_layers:

            for layer_id, info in self.feature_registry.items():

                if info["name"] == layer_name:

                    feature_outputs[layer_name] = []

                    hook = info["layer"].register_forward_hook(
                        lambda module, input, output, name=layer_name:
                            feature_outputs[name].append(output.detach())
                    )

                    hooks.append(hook)

                    break

        with torch.no_grad():

            for X, y in test_loader:

                self.model(X)

        for hook in hooks:
            hook.remove()

        for layer_name in feature_outputs:

            features[layer_name]["data"] = torch.cat(
                feature_outputs[layer_name],
                dim=0
            )

        features_dir = os.path.join(
            self.experiment_data_path,
            f"{self.experiment_name}_{self.experiment_id}",
            f"Run_{run_id:03d}",
            "features"
        )

        os.makedirs(features_dir, exist_ok=True)

        features_path = os.path.join(
            features_dir,
            "selected_model_features.pt"
        )

        torch.save(features, features_path)

        relative_features_path = os.path.relpath(
            features_path,
            self.experiment_data_path
        )

        return {
            "features_path": relative_features_path,
            "selected_model_epoch_num": selected_epoch_num
        }


#--reset the model parameters for a new run.
    def _reset_model_parameters(self):
        """Reset all model parameters for a new run."""

        for layer in self.model.model:

            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


#--give the input random_seed for Pytorch.
    def _set_random_seed(self, seed):
        """Set the random seed for the current run."""

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


#--save all experiment results to the experiment result file.
    def _save_experiment_results(self, all_runs_info):
        """Save all run results of the experiment."""

        experiment_dir = os.path.join(
            self.experiment_data_path,
            f"{self.experiment_name}_{self.experiment_id}"
        )

        os.makedirs(experiment_dir, exist_ok=True)

        experiment_results_path = os.path.join(
            experiment_dir,
            "ExperimentResults.pt"
        )

        torch.save(
            all_runs_info,
            experiment_results_path
        )


#--totally operate train process with all run times .

    def _total_train(self):
        """Train the model for all runs."""

        all_runs_info = []
        random_seed = self.base_seed

        #--build the DataLoaders for the experiment.
        train_loader, validation_loader, test_loader = (
            self._build_dataloader()
        )

        #--get every run information
        for run_id in range(1, self.runs_num + 1):

            self._set_random_seed(random_seed)

            #--reset model parameters for the current run.
            self._reset_model_parameters()

            #--get every epoch information and selected epoch num.
            epoch_info_list, selected_epoch_num = self._train_one_run(
                train_loader,
                validation_loader,
                run_id
            )

            #--get the selected model information.
            model_info = self._get_model_info(
                run_id,
                selected_epoch_num
            )

            #--get the selected model features.
            features_info = self._get_features_info(
                test_loader,
                run_id,
                selected_epoch_num
            )

            #--get the test information.
            test_info = self._test_one_run(
                test_loader
            )

            #--get the experiment information for every run.
            exper_info = {
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "run_id": run_id,
                "random_seed": random_seed
            }

            #--collect all information to create ExperimentResultContainer objective.
            run_result = ExperimentResultContainer(
                exper_info=exper_info,
                epoch_info_list=epoch_info_list,
                test_info=test_info,
                features_info=features_info,
                model_info=model_info,
                additional_info=None
            )

            all_runs_info.append(run_result)
            random_seed += 1

        #--save all experiment results.
        self._save_experiment_results(all_runs_info)

        return all_runs_info


#--Run the complete training experiment.
    def run(self):
        """Run the complete training experiment."""

        results = self._total_train()

        return results