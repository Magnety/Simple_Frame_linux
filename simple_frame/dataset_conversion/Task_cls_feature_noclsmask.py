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


import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from simple_frame.paths import raw_data

import os
#outpath = "G:/simple_frame_data_raw_base/simple_frame_raw_data"
if __name__ == "__main__":
    """
    This is the KiTS dataset after Nick fixed all the labels that had errors. Downloaded on Jan 6th 2020    
    """

    base = "/home/ubuntu/liuyiyao/Simple_Frame_data/breast_data_153_noclsmask_resample_out_spline"

    task_id = 116
    task_name = "Breast_c_noclsmask_153_reample"
    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base =  raw_data+"/"+foldername
    imagestr =  out_base+"/"+ "imagesTr"
    imagests =  out_base+"/"+"imagesTs"
    labelstr =  out_base+"/"+ "labelsTr"
    classestr = out_base+"/"+ "classesTr"
    #featurestr = out_base+"/"+ "featuresTr"
    if not os.path.isdir(imagestr):
        os.makedirs(imagestr)
    if not os.path.isdir(imagests):
        os.makedirs(imagests)
    if not os.path.isdir(labelstr):
        os.makedirs(labelstr)
    if not os.path.isdir(classestr):
        os.makedirs(classestr)
    #if not os.path.isdir(featurestr):
        #os.makedirs(featurestr)
    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)

    train_patients = all_cases[:153]
    test_patients = all_cases[153:]

    for p in train_patients:
        curr =  base+"/"+ p
        label_file =  curr+"/"+ "segmentation.nii.gz"
        image_file = curr+"/"+"imaging.nii.gz"
        class_file = curr+"/"+"label.txt"
        feature_file = curr+"/"+"feature.npy"
        shutil.copy(image_file,  imagestr+"/"+ p + "_0000.nii.gz")
        shutil.copy(label_file,  labelstr+"/"+ p + ".nii.gz")
        shutil.copy(class_file,  classestr+"/"+ p + ".txt")
        #shutil.copy(feature_file,  featurestr+"/"+ p + ".npy")
        train_patient_names.append(p)

    for p in test_patients:
        curr =  base+"/"+ p
        image_file = curr+"/"+ "imaging.nii.gz"
        shutil.copy(image_file,  imagests+"/"+ p + "_0000.nii.gz")
        test_patient_names.append(p)

    json_dict = {}
    json_dict['name'] = "Breast"
    json_dict['description'] = "Breast tumor segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "Breast data for simple_frame"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "ADC",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Tumor"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict,  out_base+"/"+ "dataset.json")
