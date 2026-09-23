#-----------------------------------------------------------------------
#--total control script for the complete DataProcessPart pipeline.
#-----------------------------------------------------------------------


from DataProcessPart.Loaders.CWRU_Loader import CWRU_Loader
from DataProcessPart.Group_Processor import Group_Processor
from DataProcessPart.Slice_Processor import Slice_Processor
from DataProcessPart.Sample_Processor import Sample_Processor


#---------------------------data processor configurations---------------------------

RAW_DATA_ROOT_PATH = r"D:\Project\PHM_Data\RawData"
DATA_SET_NAME = "CWRU"
CURRENT_DATA_PATH = r"CWRU"

#-----------------------------------------------------------------------------------


#---------------------------1. Data Loader------------------------------------------

loader = CWRU_Loader(
    RAW_DATA_ROOT_PATH,
    DATA_SET_NAME,
    CURRENT_DATA_PATH,
)

loaded_data = loader.run()


#---------------------------2. Group Processor--------------------------------------

group_processor = Group_Processor(
    loaded_data
)

grouped_data = group_processor.run()




#---------------------------3. Slice Processor--------------------------------------

slice_processor = Slice_Processor(
    grouped_data
)

sliced_data = slice_processor.run()




#---------------------------4. Sample Processor-------------------------------------

sample_processor = Sample_Processor(
    sliced_data
)

sample_data = sample_processor.run()

