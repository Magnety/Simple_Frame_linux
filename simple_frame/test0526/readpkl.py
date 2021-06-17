
"""from batchgenerators.utilities.file_and_folder_operations import *
pkl = load_pickle("/home/ubuntu/liuyiyao/Simple_Frame_data_raw_base/preprocessed/Task611_Breast_c_noclsmask_153/Plansv2.1_plans_3D.pkl")
print(pkl)
"""
import numpy as np
data = np.load("/home/ubuntu/liuyiyao/Simple_Frame_data_raw_base/preprocessed/Task611_Breast_c_noclsmask_153/plans_v2.1_stage0/case_00003.npy")
print(data.shape)