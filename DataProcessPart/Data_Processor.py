from RawDataLoader import RawDataLoader

from DataProcessPart.Loaders.CWRULoader import CWRULoader

from DataProcessPart.Group_Processor import Group_Processor

from DataProcessPart.Slice_Processor import Slice_Processor

from DataProcessPart.Sample_Processor import Sample_Processor

import time
from pathlib import Path


"""
These are configurations for all data processing steps.

And this Data_Processor.py document is used to control the data processing flow, including loading, grouping, slicing, and sampling.
"""

#--------------------------------------------------
## Configurations
#--------------------------------------------------

ROOT_DIR = r"D:\Project\PHM_Data\RawData\CWRU"     # the root directory of the CWRU dataset.

GROUP_SIZE = 10000                                #-- the size of each group when splitting the raw data into groups.

SLICE_SIZE = 1024                                 #-- the size of each slice when splitting the grouped data.

SLICE_STRIDE = 512                                #-- the stride of each slice when splitting the grouped data.

#--the configs for sample processing are a lot, so we remain it use this default config.

TRAIN_CONDITIONS = [1797, 1772, 1750]             #-- the conditions for training samples.

VALIDATION_CONDITIONS = [1730]                    #-- the conditions for validation samples.

TEST_CONDITIONS = [1730]                          #-- the conditions for testing samples.

TRAIN_SIZE = 4000                                 #-- the number of training samples.

VALIDATION_SIZE = 1000                            #-- the number of validation samples.

TEST_SIZE = 1000                                  #-- the number of testing samples.

EXCLUDED_FAULT_SIZES = ["028"]                    #-- the fault sizes to be excluded from the samples.

OR_FAULT_POSITION = 6                             #-- the fault position for OR label(3/6/12).

RANDOM_SEED = 42                                  #-- the random seed for reproducibility.

OUTPUT_DIR = r"D:\Project\PHM_Data\SampledData"   #-- the output directory for the processed data.

OUTPUT_FOLDER = "10000-1024-512-ex028-10Lables"   #-- the output folder for the processed data.


#--------------------------------------------------
#-- Start processing
#--------------------------------------------------

start_time = time.perf_counter()

print("=" * 50)
print("PHM Benchmark Data Processing")
print("=" * 50)


#--------------------------------------------------
#-- choose the loader for the raw data, here we use CWRULoader as the loader.
#--------------------------------------------------

cwru_loader = CWRULoader(
    root_dir=ROOT_DIR
)

#--RawDataLoader is a loader controller, which can be used to load raw data from different datasets by selecting the appropriate loader.

raw_loader = RawDataLoader(loader=cwru_loader)

raw_data = raw_loader.load()


print("\n[Raw Data]")
print(f"Raw records: {len(raw_data)}")


#--------------------------------------------------
#-- At this step, we use Group_Processor to split the raw data into fixed-length groups.
#--------------------------------------------------

#-- And we use the Group_Processor to slice dataset first time(First Slice).

group_processor = Group_Processor(
    group_size=GROUP_SIZE
)

grouped_data = group_processor.process(
    raw_data
)


print("\n[Grouping]")
print(f"Group size: {GROUP_SIZE}")
print(f"Groups generated: {len(grouped_data)}")


#--------------------------------------------------
#-- At this step, we use Slice_Processor to slice the grouped data into smaller segments(Second Slice).
#--------------------------------------------------

slice_processor = Slice_Processor(
    slice_size=SLICE_SIZE,
    slice_stride=SLICE_STRIDE
)

sliced_data = slice_processor.process(
    grouped_data
)


print("\n[Slicing]")
print(f"Slice size: {SLICE_SIZE}")
print(f"Slice stride: {SLICE_STRIDE}")
print(f"Slices generated: {len(sliced_data)}")


#--------------------------------------------------
#-- At this step, we use Sample_Processor to choose the samples from the sliced data for training validation and testing.
#--------------------------------------------------

sample_processor = Sample_Processor(
    train_conditions=TRAIN_CONDITIONS,
    validation_conditions=VALIDATION_CONDITIONS,
    test_conditions=TEST_CONDITIONS,
    train_size=TRAIN_SIZE,
    validation_size=VALIDATION_SIZE,
    test_size=TEST_SIZE,
    excluded_fault_sizes=EXCLUDED_FAULT_SIZES,
    or_fault_position=OR_FAULT_POSITION,
    random_seed=RANDOM_SEED,
    output_dir=OUTPUT_DIR,
    output_folder=OUTPUT_FOLDER
)

train_set, validation_set, test_set = (
    sample_processor.process(
        sliced_data
    )
)


print("\n[Sampling]")
print(f"Train samples: {len(train_set)}")
print(f"Validation samples: {len(validation_set)}")
print(f"Test samples: {len(test_set)}")


#--------------------------------------------------
#-- Output information
#--------------------------------------------------

print("\n[Output]")

output_file = Path(sample_processor.output_file)

print(f"Format: {output_file.suffix}")
print(f"File: {output_file.name}")

if output_file.exists():
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

print(f"Output path: {output_file}")


#--------------------------------------------------
#-- Processing summary
#--------------------------------------------------

total_time = time.perf_counter() - start_time

print("\n" + "=" * 50)
print("Processing Completed")
print(f"Total processing time: {total_time:.2f} s")
print("=" * 50)