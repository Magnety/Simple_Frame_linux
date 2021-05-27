
from batchgenerators.utilities.file_and_folder_operations import *
pkl = load_pickle("G:/simple_frame_data_raw_base/simple_frame_preprocessed/Task100_Breast_c_f_noclsmask/tuData_plans_v2.1_stage0/case_00000.pkl")
print(pkl)
print(len(pkl['class_locations']))