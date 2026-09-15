#---------------------------------------------------------------------
#--this file is used to load and validate experiment result data
#---------------------------------------------------------------------


import os
import torch

from ExperimentPart.ExperimentResultContainer import (
    ExperimentResultContainer,
    EpochResult
)


class ExperimentDataLoader:

    def __init__(
        self,
        experiment_name,
        experiment_id,
        experiment_data_path,
        plot_feature_layers=None
    ):

        #--the fault value for plot_feature_layers is "all"
        if plot_feature_layers is None:
            plot_feature_layers = "all"

        #--validate the experiment name to ensure it is not empty.
        if not experiment_name:
            raise ValueError(
                "ExperimentName cannot be empty. "
                "Please provide the experiment name!"
            )

        #--validate the experiment ID to ensure it is not empty.
        if experiment_id is None or str(experiment_id).strip() == "":
            raise ValueError(
                "ExperimentID cannot be empty. "
                "Please provide the experiment ID!"
    )
        #--validate the experiment_id to ensure it is a digit.
        if not str(experiment_id).isdigit():
            raise ValueError(
                "Experiment_ID is not a digit and must be a digit!"
            )

        #--format the experiment_id with the prefix "Ex"
        #--and zero-pad it to three digits.
        experiment_id = int(experiment_id)
        experiment_id = f"Ex{experiment_id:03d}"

        #--validate the experiment data path to ensure it is not empty.
        if not experiment_data_path:
            raise ValueError(
                "ExperimentDataPath cannot be empty. "
                "Please provide the experiment data save path!"
            )

        self.experiment_data_path = experiment_data_path
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id

        #--construct the specified experiment data path.
        self.experiment_path = os.path.join(
            self.experiment_data_path,
            f"{self.experiment_name}_{self.experiment_id}"
        )

        #--load the experiment result data from experiment path.
        self.experiment_results = self._load_experiment_results()

        #--validate the form and content of the experiment result data.
        self._validate_experiment_results()
        self.available_feature_layers = (
        self._get_available_feature_layers()
        )

        self.plot_feature_layers = (
            self._validate_plot_feature_layers(
                plot_feature_layers
            )

        )

    #--load the experiment result data.
    def _load_experiment_results(self):

        #--construct the experiment result file path.
        experiment_results_path = os.path.join(
            self.experiment_path,
            "ExperimentResults.pt"
        )

        #--validate if the experiment result file exists.
        if not os.path.isfile(experiment_results_path):
            raise FileNotFoundError(
                f"ExperimentResults.pt was not found: "
                f"{experiment_results_path}"
            )

        #--load the experiment result data.
        experiment_results = torch.load(
            experiment_results_path,
            weights_only=False
        )

        return experiment_results


    #--get the available feature layers from the experiment result data.
    def _get_available_feature_layers(self):

        run_result = self.experiment_results[0]

        features_info = run_result.features_info

        if features_info is None:
            raise ValueError(
                "Feature information is missing."
            )

        visualization_features_path = (
            features_info["converted_features_path"]
        )

        visualization_features_file_path = os.path.join(
            self.experiment_data_path,
            visualization_features_path
        )

        if not os.path.exists(
            visualization_features_file_path
        ):
            raise FileNotFoundError(
                "Visualization feature file not found: "
                f"{visualization_features_file_path}"
            )

        visualization_features = torch.load(
            visualization_features_file_path,
            weights_only=False
        )

        return list(
            visualization_features.keys()
        )


    #--validate the plot feature layers.
    def _validate_plot_feature_layers(
        self,
        plot_feature_layers
    ):

        print(
            "Available feature layers for visualization:"
        )

        for feature_layer in self.available_feature_layers:
            print(
                f"  - {feature_layer}"
            )

        #--"all" means that all available feature layers will be used for visualization.
        if plot_feature_layers == "all":

            return self.available_feature_layers

        #--plot_feature_layers must be either "all"
        #--or a list of feature layer names.
        if not isinstance(
            plot_feature_layers,
            list
        ):

            raise TypeError(
                "plot_feature_layers must be "
                "'all' or a list of feature layer names."
            )

        #--an empty list is not allowed.
        if len(plot_feature_layers) == 0:

            raise ValueError(
                "plot_feature_layers cannot be empty."
            )

        #--check whether every requested feature layer
        #--exists in the available feature layers.
        for feature_layer in plot_feature_layers:

            if feature_layer not in self.available_feature_layers:

                raise ValueError(
                    f"Feature layer '{feature_layer}' "
                    f"is not available. "
                    f"Available feature layers: "
                    f"{self.available_feature_layers}"
                )

        #--duplicate feature layer names are not allowed.
        if len(plot_feature_layers) != len(
            set(plot_feature_layers)
        ):

            raise ValueError(
                "plot_feature_layers contains "
                "duplicate feature layer names."
            )

        return plot_feature_layers


    #--validate the form and content of the experiment result data.
    def _validate_experiment_results(self):

        #--validate the experiment result data information.
        self._validate_experiment_result_info()

        #--validate the experiment information.
        self._validate_experiment_info()

        #--validate the epoch information.
        self._validate_epoch_info()

        #--validate the final test information.
        self._validate_test_info()

        #--validate the model information.
        self._validate_model_info()

        #--validate the feature information.
        self._validate_features_info()

        #--validate the additional information.
        self._validate_additional_info()


    #--validate the overall information of experiment result data.
    def _validate_experiment_result_info(self):

        #--validate the type of the experiment result data.
        if not isinstance(
            self.experiment_results,
            list
        ):
            raise TypeError(
                "ExperimentResults must be a list!"
            )

        #--validate that the experiment result list is not empty.
        if not self.experiment_results:
            raise ValueError(
                "ExperimentResults cannot be an empty list!"
            )

        #--validate every run result.
        for run_result in self.experiment_results:

            if not isinstance(
                run_result,
                ExperimentResultContainer
            ):
                raise TypeError(
                    "Every element in ExperimentResults must be "
                    "an instance of ExperimentResultContainer!"
                )

        
    #--validate the experiment information of every run.
    def _validate_experiment_info(self):

        expected_keys = {
            "experiment_id",
            "experiment_name",
            "run_id",
            "random_seed"
        }

        run_ids = []

        for run_result in self.experiment_results:

            exper_info = run_result.exper_info

            if not isinstance(exper_info, dict):
                raise TypeError(
                    "exper_info must be a dictionary!"
                )

            if set(exper_info.keys()) != expected_keys:
                raise ValueError(
                    "exper_info must contain exactly the following keys: "
                    f"{sorted(expected_keys)}"
                )

            if not isinstance(
                exper_info["experiment_id"],
                str
            ):
                raise TypeError(
                    "exper_info['experiment_id'] must be a string!"
                )

            if not isinstance(
                exper_info["experiment_name"],
                str
            ):
                raise TypeError(
                    "exper_info['experiment_name'] must be a string!"
                )

            if not isinstance(
                exper_info["run_id"],
                int
            ):
                raise TypeError(
                    "exper_info['run_id'] must be an integer!"
                )

            if not isinstance(
                exper_info["random_seed"],
                int
            ):
                raise TypeError(
                    "exper_info['random_seed'] must be an integer!"
                )

            #--validate experiment identity.
            if exper_info["experiment_id"] != self.experiment_id:
                raise ValueError(
                    "The experiment_id in ExperimentResults does not "
                    "match the input experiment_id!"
                )

            if exper_info["experiment_name"] != self.experiment_name:
                raise ValueError(
                    "The experiment_name in ExperimentResults does not "
                    "match the input experiment_name!"
                )

            run_ids.append(
                exper_info["run_id"]
            )

        #--validate that run IDs are continuous.
        expected_run_ids = list(
            range(1, len(run_ids) + 1)
        )

        if sorted(run_ids) != expected_run_ids:
            raise ValueError(
                "Run IDs must be continuous and start from 1!"
            )


    #--validate the epoch information of every run.
    def _validate_epoch_info(self):

        expected_keys = {
            "epoch_num",
            "epoch_train_loss",
            "epoch_train_accuracy",
            "epoch_train_time",
            "epoch_validation_loss",
            "epoch_validation_accuracy"
        }

        for run_result in self.experiment_results:

            epoch_info_list = run_result.epoch_info_list

            if not isinstance(
                epoch_info_list,
                list
            ):
                raise TypeError(
                    "epoch_info_list must be a list!"
                )

            if not epoch_info_list:
                raise ValueError(
                    "epoch_info_list cannot be an empty list!"
                )

            previous_epoch_num = 0

            for epoch_result in epoch_info_list:

                if not isinstance(
                    epoch_result,
                    EpochResult
                ):
                    raise TypeError(
                        "Every element in epoch_info_list must be "
                        "an instance of EpochResult!"
                    )

                #--validate epoch number.
                if not isinstance(
                    epoch_result.epoch_num,
                    int
                ):
                    raise TypeError(
                        "EpochResult.epoch_num must be an integer!"
                    )

                if epoch_result.epoch_num != previous_epoch_num + 1:
                    raise ValueError(
                        "Epoch numbers must be continuous and start from 1!"
                    )

                previous_epoch_num = epoch_result.epoch_num

                #--validate epoch metrics.
                self._validate_optional_numeric(
                    epoch_result.epoch_train_loss,
                    "epoch_train_loss"
                )

                self._validate_optional_numeric(
                    epoch_result.epoch_train_accuracy,
                    "epoch_train_accuracy"
                )

                self._validate_optional_numeric(
                    epoch_result.epoch_train_time,
                    "epoch_train_time"
                )

                self._validate_optional_numeric(
                    epoch_result.epoch_validation_loss,
                    "epoch_validation_loss"
                )

                self._validate_optional_numeric(
                    epoch_result.epoch_validation_accuracy,
                    "epoch_validation_accuracy"
                )


    #--validate the final test information of every run.
    def _validate_test_info(self):

        expected_keys = {
            "test_loss",
            "test_accuracy",
            "y_true",
            "y_pred",
            "label_to_index"
        }

        for run_result in self.experiment_results:

            test_info = run_result.test_info

            if not isinstance(
                test_info,
                dict
            ):
                raise TypeError(
                    "test_info must be a dictionary!"
                )

            if set(test_info.keys()) != expected_keys:
                raise ValueError(
                    "test_info must contain exactly the following keys: "
                    f"{sorted(expected_keys)}"
                )

            if not isinstance(
                test_info["test_loss"],
                (int, float)
            ):
                raise TypeError(
                    "test_info['test_loss'] must be numeric!"
                )

            if not isinstance(
                test_info["test_accuracy"],
                (int, float)
            ):
                raise TypeError(
                    "test_info['test_accuracy'] must be numeric!"
                )

            if not isinstance(
                test_info["y_true"],
                list
            ):
                raise TypeError(
                    "test_info['y_true'] must be a list!"
                )

            if not isinstance(
                test_info["y_pred"],
                list
            ):
                raise TypeError(
                    "test_info['y_pred'] must be a list!"
                )

            if not isinstance(
                test_info["label_to_index"],
                dict
            ):
                raise TypeError(
                    "test_info['label_to_index'] must be a dictionary!"
                )

            if len(test_info["y_true"]) != len(
                test_info["y_pred"]
            ):
                raise ValueError(
                    "The lengths of y_true and y_pred must be equal!"
                )


    #--validate the feature information of every run.
    def _validate_features_info(self):
        """Validate the features information of every experiment result."""

        expected_keys = {
            "converted_features_path",
            "selected_model_epoch_num"
        }

        for run_result in self.experiment_results:

            features_info = run_result.features_info
            model_info = run_result.model_info

            if not isinstance(
                features_info,
                dict
            ):
                raise TypeError(
                    "features_info must be a dictionary!"
                )

            if set(features_info.keys()) != expected_keys:
                raise ValueError(
                    "features_info must contain exactly the following keys: "
                    f"{sorted(expected_keys)}"
                )


            if not isinstance(
                features_info["converted_features_path"],
                str
            ):
                raise TypeError(
                    "features_info['converted_features_path'] "
                    "must be a string!"
                )

            if not isinstance(
                features_info["selected_model_epoch_num"],
                int
            ):
                raise TypeError(
                    "features_info['selected_model_epoch_num'] "
                    "must be an integer!"
                )

            #--validate consistency with model information.
            if (
                features_info["selected_model_epoch_num"]
                != model_info["selected_model_epoch_num"]
            ):
                raise ValueError(
                    "The selected_model_epoch_num in "
                    "features_info and model_info must be the same!"
                )


    #--validate the model information of every run.
    def _validate_model_info(self):

        expected_keys = {
            "checkpoint_path",
            "selected_model_epoch_num"
        }

        for run_result in self.experiment_results:

            model_info = run_result.model_info

            if not isinstance(
                model_info,
                dict
            ):
                raise TypeError(
                    "model_info must be a dictionary!"
                )

            if set(model_info.keys()) != expected_keys:
                raise ValueError(
                    "model_info must contain exactly the following keys: "
                    f"{sorted(expected_keys)}"
                )

            if not isinstance(
                model_info["checkpoint_path"],
                str
            ):
                raise TypeError(
                    "model_info['checkpoint_path'] must be a string!"
                )

            if not isinstance(
                model_info["selected_model_epoch_num"],
                int
            ):
                raise TypeError(
                    "model_info['selected_model_epoch_num'] "
                    "must be an integer!"
                )


    #--validate the additional information of every run.
    def _validate_additional_info(self):

        for run_result in self.experiment_results:

            additional_info = run_result.additional_info

            if (
                additional_info is not None
                and not isinstance(additional_info, dict)
            ):
                raise TypeError(
                    "additional_info must be a dictionary or None!"
                )


    #--validata the epoch num from every run is the same，and if is the same with input parameter "EPOCH". 
    #--def _validate_epoch_num_consistency(self):
    #--------------------------------

    #--validate an optional numeric value.
    def _validate_optional_numeric(
        self,
        value,
        field_name
    ):

        if value is not None and not isinstance(
            value,
            (int, float)
        ):
            raise TypeError(
                f"{field_name} must be numeric or None!"
            )


    #--construct the epoch-level view of experiment results.
    def _get_epoch_level_view(self):

        #--get the maximum epoch number among all runs.
        max_epoch_num = max(
            epoch_info.epoch_num
            for run_result in self.experiment_results
            for epoch_info in run_result.epoch_info_list
        )

        epoch_level_view = []

        for epoch_num in range(1, max_epoch_num + 1):

            epoch_info = {
                "epoch_num": epoch_num,
                "runs": []
            }

            for run_result in self.experiment_results:

                #--find the epoch information for the current run.
                current_epoch_info = next(
                    (
                        epoch_info_item
                        for epoch_info_item
                        in run_result.epoch_info_list
                        if epoch_info_item.epoch_num == epoch_num
                    ),
                    None
                )

                #--check whether the current epoch contains valid metric information.
                if (
                    current_epoch_info.epoch_train_loss is None
                    and current_epoch_info.epoch_train_accuracy is None
                    and current_epoch_info.epoch_train_time is None
                    and current_epoch_info.epoch_validation_loss is None
                    and current_epoch_info.epoch_validation_accuracy is None
                ):

                    epoch_info["runs"].append({
                        "run_id":
                            run_result.exper_info["run_id"],
                        "is_exist": False,
                        "train_loss": None,
                        "train_accuracy": None,
                        "train_time": None,
                        "validation_loss": None,
                        "validation_accuracy": None
                    })

                    continue

                #--the current run contains this epoch.
                epoch_info["runs"].append({
                    "run_id":
                        run_result.exper_info["run_id"],
                    "is_exist": True,

                    "train_loss":
                        current_epoch_info.epoch_train_loss,

                    "train_accuracy":
                        current_epoch_info.epoch_train_accuracy,

                    "train_time":
                        current_epoch_info.epoch_train_time,

                    "validation_loss":
                        current_epoch_info.epoch_validation_loss,

                    "validation_accuracy":
                        current_epoch_info.epoch_validation_accuracy
                })

            epoch_level_view.append(
                epoch_info
            )

        return epoch_level_view


    #--construct the run-level view of experiment results.
    def _get_run_level_view(self):

        run_level_view = []

        #--construct the information of every run.
        for run_result in self.experiment_results:

            exper_info = run_result.exper_info
            test_info = run_result.test_info
            features_info = run_result.features_info
            model_info = run_result.model_info

            run_num = exper_info["run_id"]

            selected_model_epoch_num = (
                model_info["selected_model_epoch_num"]
            )

            #--construct the epoch information of the current run.
            epoch_info = []

            for epoch_result in run_result.epoch_info_list:

                if (
                    epoch_result.epoch_train_loss is None
                    or epoch_result.epoch_train_accuracy is None
                    or epoch_result.epoch_train_time is None
                    or epoch_result.epoch_validation_loss is None
                    or epoch_result.epoch_validation_accuracy is None
                ):

                    current_epoch_info = {
                        "epoch_num": epoch_result.epoch_num,
                        "train_accuracy": None,
                        "train_loss": None,
                        "validation_accuracy": None,
                        "validation_loss": None,
                        "train_time": None
                    }

                else:

                    current_epoch_info = {
                        "epoch_num": epoch_result.epoch_num,
                        "train_accuracy":
                            epoch_result.epoch_train_accuracy,
                        "train_loss":
                            epoch_result.epoch_train_loss,
                        "validation_accuracy":
                            epoch_result.epoch_validation_accuracy,
                        "validation_loss":
                            epoch_result.epoch_validation_loss,
                        "train_time":
                            epoch_result.epoch_train_time
                    }

                epoch_info.append(
                    current_epoch_info
                )

            #--find the training time of the selected model epoch.
            selected_epoch_train_time = None

            for epoch_result in run_result.epoch_info_list:

                if (
                    epoch_result.epoch_num
                    == selected_model_epoch_num
                ):
                    selected_epoch_train_time = (
                        epoch_result.epoch_train_time
                    )
                    break

            #--construct the run-level information.
            run_info = {
                "experiment_name": exper_info["experiment_name"],
                "experiment_id": exper_info["experiment_id"],
                "run_num": run_num,
                "random_seed": exper_info["random_seed"],
                "selected_model_epoch_num":
                    selected_model_epoch_num,
                "selected_epoch_train_time":
                    selected_epoch_train_time,
                "epoch_info": epoch_info,
                "test_loss":
                    test_info["test_loss"],
                "test_accuracy":
                    test_info["test_accuracy"],
                "y_true": test_info["y_true"],
                "y_pred": test_info["y_pred"],
                "label_to_index": test_info["label_to_index"],
                "visualization_features_path":
                    features_info["converted_features_path"],
                "model_checkpoint_path":
                    model_info["checkpoint_path"],
                "additional_info": run_result.additional_info
            }

            run_level_view.append(
                run_info
            )

        return run_level_view


    #--run overall functions and get result doc.
    def run(self):

        epoch_level_view = self._get_epoch_level_view()
        run_level_view = self._get_run_level_view()

        return (
            epoch_level_view,
            run_level_view
        )


