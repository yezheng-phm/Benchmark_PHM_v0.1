#---------------------------------------------------------------
#--this file is used to load models and trainers.
#---------------------------------------------------------------


from ExperimentPart.Models.Model_CNN import Model_CNN
from ExperimentPart.Trainers.Trainer_CNN import Trainer_CNN

class Model_Loader:
    """
    Load the model and trainer for the experiment.
    """

    def __init__(
        self,
        model_name,
        file_path,
        num_classes,

        batch_size,
        epochs,
        runs_num,
        learning_rate,
        optimizer,
        criterion,
        base_seed,
        features_save_layers,

        experiment_name,
        experiment_id,
        experiment_data_path
    ):
        self.model_name = model_name
        self.file_path = file_path
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.epochs = epochs
        self.runs_num = runs_num
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion
        self.base_seed = base_seed
        self.features_save_layers = features_save_layers

        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.experiment_data_path = experiment_data_path



#--used to load models
    def load_model(self):
        """Load and instantiate the selected model."""

        if self.model_name == "CNN":

            model = Model_CNN(
                file_path=self.file_path,
                num_classes=self.num_classes
            )

            return model

        raise ValueError(
            f"Unsupported model: {self.model_name}"
        )


#--used to load trainers
    def load_trainer(self, model):
        """Load and instantiate the selected trainer."""

        if self.model_name == "CNN":

            trainer = Trainer_CNN(
                model=model,
                batch_size=self.batch_size,
                epochs=self.epochs,
                runs_num=self.runs_num,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                criterion=self.criterion,
                base_seed=self.base_seed,
                experiment_name=self.experiment_name,
                experiment_id=self.experiment_id,
                experiment_data_path=self.experiment_data_path,
                features_save_layers=self.features_save_layers
            )

            return trainer

        raise ValueError(
            f"Unsupported trainer for model: {self.model_name}"
        )



#--load the model and trainer that have be created.
    def load(self):
        """Load the model and trainer."""

        model = self.load_model()

        trainer = self.load_trainer(model)

        return model, trainer


#--used to run the experiment
    def run(self):
        """Run the experiment."""

        model, trainer = self.load()

        results = trainer.run()

        return results