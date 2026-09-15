#---------------------------------------------------------
#--this file is used to test the Evaluation module.
#---------------------------------------------------------

import os

from EvaluationPart.ExperimentDataLoader import (
    ExperimentDataLoader
)

from EvaluationPart.Evaluation_processor import (
    Evaluation
)


data_loader = ExperimentDataLoader(
    experiment_name="CNN_Test",
    experiment_id="1",
    experiment_data_path=r"D:\Project\PHM_Data\ExperimentData\tests"
)


evaluation = Evaluation(
    data_loader
)


evaluation_results = evaluation.run()

