#---------------------------------------------------------
#--this file is a trainer for the Model_Transformer,
#--which is used to train the Transformer model
#--on the given dataset.
#---------------------------------------------------------

from datetime import datetime
import os
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DataAdaptationPart.TransformerDataAdapter import TransformerDataAdapter
from ExperimentPart.ExperimentResultContainer import (
    ExperimentResultContainer,
    EpochResult
)
from ExperimentPart.Models.Model_Transformer import Model_Transformer


class Trainer_Transformer:
    """
    A trainer class for the Model_Transformer model.
    """

    def __init__(
        self,
        file_path,
        batch_size,
        epochs,
        runs_num,
        num_classes,
        learning_rate,
        optimizer,
        criterion,
        base_seed,
        experiment_name,
        experiment_id,
        experiment_data_path,
        features_save_layers = None
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.runs_num = runs_num
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_instance = None
        self.criterion = criterion
        self.criterion_instance = None
        self.base_seed = base_seed
        self.num_classes = num_classes

        #--create transformer adapter.
        self.data_adapter = TransformerDataAdapter(
            file_path
        )
        #--set the default feature layers to save.
        if features_save_layers is None:
            features_save_layers = ["all"]

        self.features_save_layers = features_save_layers

        #--validate the basic experiment inputs.
        (
            self.experiment_name,
            self.experiment_id,
            self.experiment_data_path
        ) = self._validate_basic_inputs(
            experiment_name,
            experiment_id,
            experiment_data_path
        )



#--------------------------------external method logic area----------------------------------------------

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

    
    #--validate the basic experiment inputs.
    def _validate_basic_inputs(
        self,
        experiment_name,
        experiment_id,
        experiment_data_path
    ):
        """Validate and standardize the basic experiment inputs."""

        #--validate the experiment ID.
        if not str(experiment_id).isdigit():
            raise ValueError(
                "Experiment_ID is not a digit and must be a digit!"
            )

        #--validate the experiment name.
        if not experiment_name:
            raise ValueError(
                "Experiment_Name cannot be empty. "
                "Please provide the experiment name!"
            )

        #--validate the experiment data path.
        if not experiment_data_path:
            raise ValueError(
                "ExperimentDataPath cannot be empty. "
                "Please provide the experiment data save path!"
            )

        #--format the experiment ID.
        experiment_id = int(experiment_id)
        experiment_id = f"Ex{experiment_id:03d}"

        return (
            experiment_name,
            experiment_id,
            experiment_data_path
        )


    #--get the layer names from the input Transformer model.
    def _get_model_layer_name_list(self):
        """Get the valid feature layer names from the Transformer model."""

        #--create the layer name list.
        model_layer_name_list = [
            "input projection"
        ]

        #--get the number of Transformer Encoder layers.
        num_encoder_layers = len(
            self.model.transformer_encoder.layers
        )

        #--add the name of every Encoder layer.
        for layer_index in range(num_encoder_layers):

            model_layer_name_list.append(
                f"encoder{layer_index + 1}"
            )

        #--add the global pooling and classifier layers.
        model_layer_name_list.extend([
            "globalpool",
            "classifier",
            "all"
        ])

        return model_layer_name_list


    #--validate the input feature layer names according to
    #--the Transformer layer name protocol.
    def _validate_input_layers_name_protocol(self, input_layers):
        """Validate and standardize the input feature layer names."""

        #--validate the input type.
        if not isinstance(input_layers, list):
            raise TypeError(
                "Features_Save_Layers must be a list."
            )

        standardized_layers = []

        #--validate every input layer name.
        for layer_name in input_layers:

            #--validate the layer name type.
            if not isinstance(layer_name, str):
                raise TypeError(
                    "Each feature layer name must be a string."
                )

            #--standardize the layer name.
            layer_name = layer_name.lower()

            #--validate whether the input layer name
            #--exists in the actual model layer name list.
            if layer_name not in self.model_layer_name_list:
                raise ValueError(
                    f"Invalid feature layer name: '{layer_name}'. "
                    f"Available feature layers: "
                    f"{self.model_layer_name_list}"
                )

            standardized_layers.append(layer_name)

        #--remove duplicated layer names while preserving their order.
        standardized_layers = list(
            dict.fromkeys(standardized_layers)
        )

        return standardized_layers


    #--create DataLoaders for the training, validation, and test datasets.
    def _build_dataloader(self):
        """Build DataLoaders for training, validation, and test datasets."""

        train_dataset = TensorDataset(
            self.data_adapter.train_X,
            self.data_adapter.train_y
        )

        validation_dataset = TensorDataset(
            self.data_adapter.validation_X,
            self.data_adapter.validation_y
        )

        test_dataset = TensorDataset(
            self.data_adapter.test_X,
            self.data_adapter.test_y
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

        return (
            train_loader,
            validation_loader,
            test_loader
        )


    #--create a method for showing process
    def _create_epoch_progress(
            self,
            train_loader,
            epoch,
            run_id
        ):
            return tqdm(
                train_loader,
                desc=f"Run [{run_id}] | Epoch [{epoch}/{self.epochs}]",
                leave=True,
                colour="green",
                bar_format="{desc} |{bar:50}| {percentage:3.0f}%",
                ascii="□■"
            )

    #--create a method to train the model for one epoch.
    def _train_one_epoch(self, train_loader, epoch, run_id):
        """Train the model for one epoch."""

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        #--start the timer for the epoch training time.
        start_time = time.time()

        epoch_progress = self._create_epoch_progress(
            train_loader,
            epoch,
            run_id
        )

        for X, y in epoch_progress:

            self.optimizer_instance.zero_grad()

            output = self.model(X)

            loss = self.criterion_instance(
                output,
                y
            )

            loss.backward()

            self.optimizer_instance.step()

            total_loss += loss.item() * X.size(0)

            predicted = output.argmax(dim=1)

            correct += (
                predicted == y
            ).sum().item()

            total += y.size(0)

            epoch_loss = total_loss / total
            epoch_accuracy = correct / total

            epoch_progress.set_postfix(
                acc=f"{epoch_accuracy:.3f}",
                loss=f"{epoch_loss:.3f}"
            )

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        #--get the training time for one epoch.
        epoch_train_time = time.time() - start_time

        return (
            epoch_loss,
            epoch_accuracy,
            epoch_train_time
        )


    #--verify the model on the validation dataset for one epoch.
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


    #--core train logic for training the model for one run
    #--and selecting the best epoch based on validation performance.
    def _train_one_run(
            self,
            train_loader,
            validation_loader,
            run_id
        ):

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

        #--record the training start time.
        train_start_time = datetime.now()

        print("=" * 60)
        print(
            f"Run [{run_id}] Training Started: "
            f"{train_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):

            epoch_train_loss, epoch_train_accuracy, epoch_train_time = (
                self._train_one_epoch(
                    train_loader,
                    epoch,
                    run_id
                )
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

            #--select the first epoch as the initial best epoch.
            if selected_epoch_num is None:

                selected_epoch_num = epoch
                best_validation_accuracy = epoch_validation_accuracy
                best_validation_loss = epoch_validation_loss
                best_epoch_train_time = epoch_train_time

                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

            #--select the current epoch when validation accuracy is better.
            elif epoch_validation_accuracy > best_validation_accuracy:

                selected_epoch_num = epoch
                best_validation_accuracy = epoch_validation_accuracy
                best_validation_loss = epoch_validation_loss
                best_epoch_train_time = epoch_train_time

                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

            #--when validation accuracy is equal,
            #--select the epoch with lower validation loss.
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

            #--when validation accuracy and loss are equal,
            #--select the epoch with shorter training time.
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

        #--restore the selected best model state.
        self.model.load_state_dict(
            best_model_state
        )

        #--record the training end time.
        train_end_time = datetime.now()

        #--calculate the total training time in minutes.
        train_duration = train_end_time - train_start_time
        train_duration_minutes = round(
            train_duration.total_seconds() / 60
        )

        print("-" * 60)
        print(
            f"Run [{run_id}] Training Finished: "
            f"{train_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            f"Run [{run_id}] Training Time: "
            f"{train_duration_minutes} min"
        )
        print("=" * 60)

        return epoch_info_list, selected_epoch_num


    #--core train logic for testing the model for one run.
    def _test_one_run(self, test_loader):
        """Test the model using the selected best model."""

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        y_true = []
        y_pred = []

        #--store the raw features temporarily for feature extraction.
        raw_features = {}

        with torch.no_grad():

            for X, y in test_loader:

                #--input projection.
                x = self.model.input_projection(X)

                raw_features.setdefault(
                    "input projection",
                    []
                ).append(
                    x.cpu()
                )

                #--add positional encoding.
                x = x + self.model.positional_encoding

                #--Transformer encoder.
                for layer_index, encoder_layer in enumerate(
                    self.model.transformer_encoder.layers
                ):

                    x = encoder_layer(x)

                    layer_name = f"encoder{layer_index + 1}"

                    raw_features.setdefault(
                        layer_name,
                        []
                    ).append(
                        x.cpu()
                    )

                #--global average pooling.
                x = self.model._global_average_pooling(x)

                raw_features.setdefault(
                    "globalpool",
                    []
                ).append(
                    x.cpu()
                )

                #--classifier.
                output = self.model.classifier(x)

                raw_features.setdefault(
                    "classifier",
                    []
                ).append(
                    output.cpu()
                )

                #--calculate test loss and accuracy.
                loss = self.criterion_instance(
                    output,
                    y
                )

                total_loss += loss.item() * X.size(0)

                predicted = output.argmax(dim=1)

                correct += (
                    predicted == y
                ).sum().item()

                total += y.size(0)

                y_true.extend(
                    y.cpu().tolist()
                )

                y_pred.extend(
                    predicted.cpu().tolist()
                )

        #--combine the raw features from all batches.
        for layer_name in raw_features:

            raw_features[layer_name] = torch.cat(
                raw_features[layer_name],
                dim=0
            )

        #--temporarily store the raw features for _get_features_info().
        self.raw_features = raw_features

        test_loss = total_loss / total
        test_accuracy = correct / total

        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "y_true": y_true,
            "y_pred": y_pred,
            "label_to_index": self.data_adapter.label_to_index
        }


    #--get the mapping for every layer according to the model .
    def _get_feature_registry(self):
        """Build the feature registry for the Transformer model."""

        feature_registry = {}

        feature_registry["input projection"] = (
            self.model.input_projection
        )

        for layer_index, encoder_layer in enumerate(
            self.model.transformer_encoder.layers
        ):
            feature_registry[
                f"encoder{layer_index + 1}"
            ] = encoder_layer

        feature_registry["globalpool"] = (
            self.model._global_average_pooling
        )

        feature_registry["classifier"] = (
            self.model.classifier
        )

        return feature_registry


    #--convert selected features into representations suitable for feature visualization.
    def _convert_features_for_visualization(self, features):
        """Convert Transformer features into representations suitable for feature visualization."""

        converted_features = {}

        for layer_name, feature in features.items():

            if (
                layer_name == "input projection"
                or layer_name.startswith("encoder")
            ):
                converted_features[layer_name] = (
                    feature.mean(dim=1)
                )

            else:
                converted_features[layer_name] = feature

        return converted_features


    #--get the features from the model layers from selected layer.
    def _get_features_info(
        self,
        run_id,
        selected_epoch_num
    ):
        """Convert and save the selected model features."""

        #--get the raw features temporarily stored by _test_one_run().
        raw_features = self.raw_features

        #--determine the feature layers to save.
        if "all" in self.features_save_layers:
            feature_layers = [
                layer_name
                for layer_name in self.feature_registry
            ]
        else:
            feature_layers = self.features_save_layers.copy()

        #--select the required raw features according to the feature layers.
        selected_features = {}

        for layer_name in feature_layers:

            selected_features[layer_name] = (
                raw_features[layer_name]
            )

        #--convert the selected features for visualization.
        converted_features = (
            self._convert_features_for_visualization(
                selected_features
            )
        )

        #--create the feature save directory.
        features_dir = os.path.join(
            self.experiment_data_path,
            f"{self.experiment_name}_{self.experiment_id}",
            "Features",
            f"Run_{run_id:03d}"
        )

        os.makedirs(
            features_dir,
            exist_ok=True
        )

        #--save the converted features.
        converted_features_path = os.path.join(
            features_dir,
            "converted_for_visualization_features.pt"
        )

        torch.save(
            converted_features,
            converted_features_path
        )

        #--get the relative converted feature path.
        relative_converted_features_path = os.path.relpath(
            converted_features_path,
            self.experiment_data_path
        )

        #--release the temporary raw features.
        self.raw_features = None

        return {
            "converted_features_path": relative_converted_features_path,
            "selected_model_epoch_num": selected_epoch_num
        }


    #--create a method to save the selected model of the current run and return its information.
    def _get_model_info(self, run_id, selected_epoch_num):
        """Save the selected model of the current run and return its information."""

        experiment_name = f"{self.experiment_name}_{self.experiment_id}"

        experiment_dir = os.path.join(
            self.experiment_data_path,
            experiment_name
        )

        model_dir = os.path.join(
            experiment_dir,
            "Models",
            f"Run_{run_id:03d}"
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )

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


    #--create a method to set the random seed for the current run.
    def _set_random_seed(self, random_seed):
        """Set the random seed for the current run."""

        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)


    #--create a method to save all experiment results.
    def _save_experiment_results(self, all_runs_info):
        """Save all experiment results."""

        experiment_name = f"{self.experiment_name}_{self.experiment_id}"

        experiment_dir = os.path.join(
            self.experiment_data_path,
            experiment_name
        )

        os.makedirs(
            experiment_dir,
            exist_ok=True
        )

        experiment_results_path = os.path.join(
            experiment_dir,
            "ExperimentResults.pt"
        )

        torch.save(
            all_runs_info,
            experiment_results_path
        )

        return experiment_results_path


    #--create a method to train the model for all runs.
    def _total_train(self):
        """Train the model for all runs."""

        all_runs_info = []
        random_seed = self.base_seed

        #--build the DataLoaders for the experiment.
        train_loader, validation_loader, test_loader = (
            self._build_dataloader()
        )

        #--get every run information.
        for run_id in range(1, self.runs_num + 1):

            #--set the random seed for the current run.
            self._set_random_seed(random_seed)

            #--create a new model for the current run.
            self.model = Model_Transformer(
                num_classes=self.num_classes
            )

            #--initialize feature extraction information only for the first run.
            if run_id == 1:

                #--get the layer names from the input Transformer model.
                self.model_layer_name_list = (
                    self._get_model_layer_name_list()
                )

                #--verify the input layers name that user wants to extract features.
                self.features_save_layers = (
                    self._validate_input_layers_name_protocol(
                        self.features_save_layers
                    )
                )

                #--build the feature registry.
                self.feature_registry = (
                    self._get_feature_registry()
                )

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

            #--get the test information.
            test_info = self._test_one_run(
                test_loader
            )

            #--get the selected model features.
            features_info = self._get_features_info(
                run_id,
                selected_epoch_num
            )

            #--get the experiment information for every run.
            exper_info = {
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "run_id": run_id,
                "random_seed": random_seed
            }

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

        self._save_experiment_results(
            all_runs_info
        )

        return all_runs_info


    #--Run the complete training experiment.
    def run(self):
        """Run the complete training experiment."""

        results = self._total_train()

        return results