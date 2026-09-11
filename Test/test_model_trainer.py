import os
import torch

from ExperimentPart.Models.Model_CNN import Model_CNN
from ExperimentPart.Trainers.Trainer_CNN import Trainer_CNN


# ---------------------------------------------------------
# Test Model_CNN and Trainer_CNN
# ---------------------------------------------------------

#--dataset path
data_file_path = r"D:\Project\PHM_Data\SampledData\zy_10000_1024_512_ex028_10Lables\sampled_data_20260901164611.pt"

#--experiment configuration
num_classes = 10
batch_size = 32
epochs = 3
runs_num = 2
learning_rate = 0.001
optimizer = "Adam"
criterion = "CrossEntropyLoss"
base_seed = 42

experiment_name = "model_CNN_tests"
experiment_id = 1

experiment_data_path = r"D:\Project\PHM_Data\ExperimentData\tests"

features_save_layers = [
    "conv1",
    "globalpool"
]


# ---------------------------------------------------------
# Create Model
# ---------------------------------------------------------

model = Model_CNN(
    file_path=data_file_path,
    num_classes=num_classes
)


# ---------------------------------------------------------
# Create Trainer
# ---------------------------------------------------------

trainer = Trainer_CNN(
    model=model,
    batch_size=batch_size,
    epochs=epochs,
    runs_num=runs_num,
    learning_rate=learning_rate,
    optimizer=optimizer,
    criterion=criterion,
    base_seed=base_seed,
    experiment_name=experiment_name,
    experiment_id=experiment_id,
    experiment_data_path=experiment_data_path,
    features_save_layers=features_save_layers
)


# ---------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------

all_runs_info = trainer._total_train()


# ---------------------------------------------------------
# Experiment result path
# ---------------------------------------------------------

experiment_dir = os.path.join(
    experiment_data_path,
    f"{experiment_name}_{trainer.experiment_id}"
)

experiment_result_path = os.path.join(
    experiment_dir,
    "ExperimentResults.pt"
)


# ---------------------------------------------------------
# Create test result document
# ---------------------------------------------------------

test_result_path = os.path.join(
    experiment_dir,
    "TestResult.txt"
)


with open(test_result_path, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("CNN Experiment Test Result\n")
    f.write("=" * 70 + "\n\n")


    # -----------------------------------------------------
    # Basic experiment information
    # -----------------------------------------------------

    f.write("Experiment Information\n")
    f.write("-" * 70 + "\n")
    f.write(f"Experiment Name : {trainer.experiment_name}\n")
    f.write(f"Experiment ID   : {trainer.experiment_id}\n")
    f.write(f"Runs Number     : {trainer.runs_num}\n")
    f.write(f"Epochs          : {trainer.epochs}\n")
    f.write(f"Batch Size      : {trainer.batch_size}\n")
    f.write(f"Base Seed       : {trainer.base_seed}\n")
    f.write(
        f"Features Layers : {trainer.features_save_layers}\n"
    )
    f.write("\n")


    # -----------------------------------------------------
    # Print and save all Run information
    # -----------------------------------------------------

    for run_index, run_result in enumerate(
        all_runs_info,
        start=1
    ):

        f.write("=" * 70 + "\n")
        f.write(f"Run {run_index}\n")
        f.write("=" * 70 + "\n\n")


        #--experiment information
        f.write("Experiment Info\n")
        f.write("-" * 70 + "\n")

        for key, value in run_result.exper_info.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")


        #--epoch information
        f.write("Epoch Information\n")
        f.write("-" * 70 + "\n")

        for epoch_info in run_result.epoch_info_list:

            f.write(
                f"Epoch {epoch_info.epoch_num}\n"
            )

            f.write(
                f"  Train Loss       : "
                f"{epoch_info.epoch_train_loss:.6f}\n"
            )

            f.write(
                f"  Train Accuracy   : "
                f"{epoch_info.epoch_train_accuracy:.6f}\n"
            )

            f.write(
                f"  Validation Loss  : "
                f"{epoch_info.epoch_validation_loss:.6f}\n"
            )

            f.write(
                f"  Validation Acc   : "
                f"{epoch_info.epoch_validation_accuracy:.6f}\n"
            )

            f.write(
                f"  Training Time    : "
                f"{epoch_info.epoch_train_time:.6f} s\n"
            )

            f.write("\n")


        #--test information
        f.write("Test Information\n")
        f.write("-" * 70 + "\n")

        test_info = run_result.test_info

        f.write(
            f"Test Loss         : "
            f"{test_info['test_loss']:.6f}\n"
        )

        f.write(
            f"Test Accuracy     : "
            f"{test_info['test_accuracy']:.6f}\n"
        )

        f.write(
            f"Number of Samples : "
            f"{len(test_info['y_true'])}\n"
        )

        f.write(
            f"Label Mapping     : "
            f"{test_info['label_to_index']}\n"
        )

        f.write("\n")


        #--selected model information
        f.write("Model Information\n")
        f.write("-" * 70 + "\n")

        for key, value in run_result.model_info.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")


        #--feature information
        f.write("Feature Information\n")
        f.write("-" * 70 + "\n")

        for key, value in run_result.features_info.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")


# ---------------------------------------------------------
# Print result to console
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CNN Experiment Test Result")
print("=" * 70)

for run_index, run_result in enumerate(
    all_runs_info,
    start=1
):

    print(f"\n{'=' * 70}")
    print(f"Run {run_index}")
    print(f"{'=' * 70}")

    print("\nExperiment Info:")
    print(run_result.exper_info)

    print("\nEpoch Information:")

    for epoch_info in run_result.epoch_info_list:

        print(
            f"Epoch {epoch_info.epoch_num}: "
            f"Train Loss={epoch_info.epoch_train_loss:.6f}, "
            f"Train Acc={epoch_info.epoch_train_accuracy:.6f}, "
            f"Val Loss={epoch_info.epoch_validation_loss:.6f}, "
            f"Val Acc={epoch_info.epoch_validation_accuracy:.6f}, "
            f"Time={epoch_info.epoch_train_time:.6f}s"
        )

    print("\nTest Information:")

    print(
        f"Test Loss     : "
        f"{run_result.test_info['test_loss']:.6f}"
    )

    print(
        f"Test Accuracy : "
        f"{run_result.test_info['test_accuracy']:.6f}"
    )

    print(
        f"Samples       : "
        f"{len(run_result.test_info['y_true'])}"
    )

    print(
        f"Label Mapping : "
        f"{run_result.test_info['label_to_index']}"
    )

    print("\nModel Information:")
    print(run_result.model_info)

    print("\nFeature Information:")
    print(run_result.features_info)


# ---------------------------------------------------------
# Check ExperimentResult.pt
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("Checking ExperimentResults.pt")
print("=" * 70)

if os.path.exists(experiment_result_path):

    print(
        f"ExperimentResults.pt exists: "
        f"{experiment_result_path}"
    )

    saved_results = torch.load(
        experiment_result_path,
        weights_only=False
    )

    print(
        f"Loaded result type: "
        f"{type(saved_results)}"
    )

    print(
        f"Loaded Run number: "
        f"{len(saved_results)}"
    )

else:

    print("ERROR: ExperimentResults.pt was not found.")


# ---------------------------------------------------------
# Check model and feature files
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("Checking Run Files")
print("=" * 70)

for run_id in range(1, runs_num + 1):

    run_dir = os.path.join(
        experiment_dir,
        f"Run_{run_id:03d}"
    )

    model_path = os.path.join(
        run_dir,
        "model",
        "selected_model.pt"
    )

    features_path = os.path.join(
        run_dir,
        "features",
        "selected_model_features.pt"
    )


    #--check selected model
    print(f"\nRun_{run_id:03d}")

    if os.path.exists(model_path):

        print(
            f"  selected_model.pt: OK"
        )

        model_state = torch.load(
            model_path,
            weights_only=False
        )

        print(
            f"  Model state type: "
            f"{type(model_state)}"
        )

    else:

        print(
            f"  selected_model.pt: MISSING"
        )


    #--check selected features
    if os.path.exists(features_path):

        print(
            f"  selected_model_features.pt: OK"
        )

        features = torch.load(
            features_path,
            weights_only=False
        )

        print(
            f"  Feature names: "
            f"{list(features.keys())}"
        )

        for feature_name, feature_info in features.items():

            if feature_name == "all":
                continue

            print(
                f"  {feature_name}: "
                f"feature_id={feature_info['feature_id']}, "
                f"shape={tuple(feature_info['data'].shape)}"
            )

    else:

        print(
            f"  selected_model_features.pt: MISSING"
        )


# ---------------------------------------------------------
# Test result document
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("Test Result Document")
print("=" * 70)

print(
    f"Saved to:\n{test_result_path}"
)

print("=" * 70)