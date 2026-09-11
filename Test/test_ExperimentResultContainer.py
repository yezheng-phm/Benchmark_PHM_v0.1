

from ExperimentPart.ExperimentResultContainer import ExperimentResultContainer, EpochResult




epoch_result = EpochResult(
    epoch_num=1,
    epoch_train_loss=0.5,
    epoch_train_accuracy=0.8,
    epoch_validation_loss=0.6,
    epoch_validation_accuracy=0.75
)

result = ExperimentResultContainer(
    exper_info={
        "experiment_id": "EXP_001",
        "dataset": "CWRU",
        "model_name": "CNN",
        "run_id": 1,
        "random_seed": 42
    },

    epoch_info_list=[epoch_result],

    test_info={
        "selected_epoch": 1,
        "y_true": [0, 1, 2],
        "y_pred": [0, 1, 1],
        "accuracy": 0.667,
        "loss": 0.8
    },

    features_info={
        "layer_features": {}
    },

    model_info={
        "checkpoint_path": "checkpoint.pt"
    }
)

print(result)