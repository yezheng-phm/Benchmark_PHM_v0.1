#---------------------------------------------------------------
#--this file is used to control the experiment.
#---------------------------------------------------------------


from ExperimentPart.Model_Loader import Model_Loader

if __name__ == "__main__":


    print("ExperimentControl started.")

#---------------------------------------------------------------
#--experiment configuration
#---------------------------------------------------------------

#------------------model configuration--------------------------#

    MODEL_NAME = "CNN"

    FILE_PATH = r"D:\Project\PHM_Data\SampledData\zy_10000_1024_512_ex028_10Lables\sampled_data_20260901164611.pt"
    NUM_CLASSES = 10


    #-----------------trainer configuration-------------------------#

    BATCH_SIZE = 32
    EPOCHS = 100
    RUNS_NUM = 10

    LEARNING_RATE = 0.001
    OPTIMIZER = "Adam"
    CRITERION = "CrossEntropyLoss"

    BASE_SEED = 42

    FEATURES_SAVE_LAYERS = [
        "conv1",
        "conv2",
        "conv3",
        "globalpool"
    ]

    #-----------------experiment configuration-------------------------#

    EXPERIMENT_NAME = "CNN_Test_100epochs"
    EXPERIMENT_ID = "05"
    EXPERIMENT_DATA_PATH = r"D:\Project\PHM_Data\ExperimentData\tests"


    #--create a Model_Loader objective.

    model_loader = Model_Loader(
        model_name=MODEL_NAME,
        file_path=FILE_PATH,
        num_classes=NUM_CLASSES,

        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        runs_num=RUNS_NUM,
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        base_seed=BASE_SEED,
        features_save_layers=FEATURES_SAVE_LAYERS,

        experiment_name=EXPERIMENT_NAME,
        experiment_id=EXPERIMENT_ID,
        experiment_data_path=EXPERIMENT_DATA_PATH
    )


model_loader.run()

