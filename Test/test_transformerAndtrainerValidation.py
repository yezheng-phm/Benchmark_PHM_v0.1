#---------------------------------------------------------
#--this file is used to test the Model_Transformer
#--and Trainer_Transformer.
#---------------------------------------------------------

from ExperimentPart.Models.Model_Transformer import Model_Transformer
from ExperimentPart.Trainers.Trainer_Transformer import Trainer_Transformer


if __name__ == "__main__":

    #--define the experiment data path.
    experiment_data_path = (
        r"D:\Project\PHM_Data\SampledData\10000-1024-512-ex028-10Lables\sampled_data_20260910150414.pt"
    )

    #--create the Transformer model.
    model = Model_Transformer(
        file_path=experiment_data_path,
        num_classes=10
    )

    #--create the Transformer trainer.
    trainer = Trainer_Transformer(
        model=model,
        batch_size=32,
        epochs=1,
        runs_num=1,
        learning_rate=0.001,
        optimizer="Adam",
        criterion="CrossEntropyLoss",
        base_seed=42,
        experiment_name="Transformer_Test",
        experiment_id=1,
        experiment_data_path="ExperimentResults",
        features_save_layers=[
            "Input Projection",
            "Encoder1",
            "GLOBALPOOL"
        ]
    )

    #--build the DataLoaders.
    train_loader, validation_loader, test_loader = (
        trainer._build_dataloader()
    )

    #--get one batch from each DataLoader.
    train_X, train_y = next(iter(train_loader))
    validation_X, validation_y = next(iter(validation_loader))
    test_X, test_y = next(iter(test_loader))

    #--print the batch information.
    print("\nTrain batch:")
    print("X shape:", train_X.shape)
    print("y shape:", train_y.shape)

    print("\nValidation batch:")
    print("X shape:", validation_X.shape)
    print("y shape:", validation_y.shape)

    print("\nTest batch:")
    print("X shape:", test_X.shape)
    print("y shape:", test_y.shape)