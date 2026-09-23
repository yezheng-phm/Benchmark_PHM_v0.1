#-----------------------------------------------------------------------------------
#--This file is used to create a data container.
#--All modules of DataProcess use this container for generating and consuming data.
#-----------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DataProcessContainer:

    #--used to store current form data.
    X_data: Optional[np.ndarray] = None 
    #--used to store the specified class lable.
    y: Optional[str] = None
    #--used to store the core information of the data.
    metadata: Optional[dict] = None
    #--used to store the information after grouping.
    group_info: Optional[dict] = None
    #--used to store the information after slicing.
    slice_info: Optional[dict] = None
    #--used to store the information after sampling.
    sample_info: Optional[dict] = None
    #--used to store the extra information of the data.
    extra_info: Optional[dict] = None