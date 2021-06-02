import os
input_dir = "/home/ubuntu/liuyiyao/Simple_Frame_data_raw_base/raw_data/Task100_Breast_c_f_noclsmask/classesTr"
names = os.listdir(input_dir)
for name in names:
    class_source = open(input_dir + '/' +  name)  # 打开源文件
    indate = class_source.read()  # 显示所有源文件内容
    print(name.split('.')[0],':',indate)