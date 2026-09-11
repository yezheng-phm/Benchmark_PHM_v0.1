#---------------------------------------------
# this file is used to store the experiment results, and other modules can consume the results from the output of this file.
#---------------------------------------------

from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentResultContainer:
    """
    A container class to store experiment results.
    """

    #--used for storing the core experiment information.
    #  experiment_id: used to identify ID of the experiment
    #  dataset: the dataset used in the experiment
    #  model_name: the name of the model used in the experiment
    #  run_id: the ID of the current run
    #  random_seed: the random seed used in the current run
    exper_info: dict  

    #--used for storing the core information of every epoch.
    #  every element in the list is an instance of EpochResult, which contains the training and validation loss and accuracy of the corresponding epoch.
    epoch_info_list: list["EpochResult"]

    #--used for storing the final test result.
    #  selected_epoch: the selected epoch based on validation performance.
    #  y_true: the ground-truth labels of the test dataset.
    #  y_pred: the predicted labels of the test dataset.
    #  accuracy: the test accuracy.
    #  loss: the test loss.
    test_info: dict

    #--feature extracted from the selected model version on the test dataset.
    #  layer_features: features extracted from the selected layers.
    features_info: dict

    #--the information of the model.
    #  checkpoint_path: the path of the selected model checkpoint.
    model_info: dict

    #--additional information that can be used for storing any other information.
    #  This field is optional and can be used for information extensions.
    additional_info: Optional[dict] = None


@dataclass
class EpochResult :
    """
    A container class to store the result of every single epoch.
    """

    #--the epoch number.
    epoch_num:int

    #--the training loss and accuracy of the epoch.
    epoch_train_loss:float

    epoch_train_accuracy:float

    #--the validation loss and accuracy of the epoch.
    epoch_validation_loss:float

    epoch_validation_accuracy:float

    #--the training time of the epoch.
    epoch_train_time: float