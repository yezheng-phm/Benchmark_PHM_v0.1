#---------------------------------------------------------
#--this file is used to test the Visualization module.
#---------------------------------------------------------

from EvaluationPart.ExperimentDataLoader import (
    ExperimentDataLoader
)

from EvaluationPart.Visualization_processor import (
    Visualization
)


#---------------------------------------------------------
#--create the experiment data loader.
#---------------------------------------------------------


import torch

features_path = (
    r"D:\Project\PHM_Data\ExperimentData\tests"
    r"\Transformer_Test_Ex002"
    r"\Features\Run_001"
    r"\selected_model_features.pt"
)


data_loader = ExperimentDataLoader(
    experiment_name="Transformer_Test",
    experiment_id=2,
    experiment_data_path=r"D:\Project\PHM_Data\ExperimentData\tests"
)


#---------------------------------------------------------
#--create the visualization processor.
#---------------------------------------------------------

visualization = Visualization(
    data_loader=data_loader,
    publication_output=False
)

#---------------------------------------------------------
#--get the experiment views from the data loader.
#---------------------------------------------------------

epoch_level_view, run_level_view = (
    data_loader.run()
)


#---------------------------------------------------------
#--plot the box plots.
#---------------------------------------------------------



visualization.run(
    epoch_level_view, 
    run_level_view

)

