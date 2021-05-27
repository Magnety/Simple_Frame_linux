#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "lyy"
default_plans_identifier = "lyyPlans"
default_data_identifier = 'lyyDatas'
default_trainer = "cls_trainer"
default_cascade_trainer = "tuTrainerV2CascadeFullRes"


base = "G:/Simple_Frame_data_raw_base"
preprocessing_output_dir =base+"/preprocessed"
network_training_output_dir_base =base+"/trained_models"

if base is not None:
    raw_data =base +"/raw_data"
    cropped_data = base +"/cropped_data"
    if not os.path.isdir(raw_data):os.makedirs(raw_data)
    if not os.path.isdir(cropped_data):os.makedirs(cropped_data)
else:
    print("simple_frame_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    cropped_data = raw_data = None

if preprocessing_output_dir is not None:
    if not os.path.isdir(preprocessing_output_dir):os.makedirs(preprocessing_output_dir)

else:
    print("simple_frame_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = network_training_output_dir_base+'/'+ my_output_identifier
    if not os.path.isdir(network_training_output_dir):os.makedirs(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information on how to set this "
          "up.")
    network_training_output_dir = None
