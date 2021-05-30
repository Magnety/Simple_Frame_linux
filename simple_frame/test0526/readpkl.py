
from batchgenerators.utilities.file_and_folder_operations import *
pkl = load_pickle("/home/ubuntu/liuyiyao/Simple_Frame_data_raw_base/preprocessed/Task100_Breast_c_f_noclsmask/plans_v2.1_stage0/case_00000.pkl")
print(pkl['connect_mask_box'].keys())
for i in pkl['connect_mask_box'].keys():
    print(pkl['connect_mask_box'][i][0][0])