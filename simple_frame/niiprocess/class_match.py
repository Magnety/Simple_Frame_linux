import os
dir = "/home/ubuntu/liuyiyao/Simple_Frame_data/breast_data_153_noclsmask"
dir1="/home/ubuntu/liuyiyao/Simple_Frame_data"
names = os.listdir(dir)
j=0
for name in names:
    class_source = open(dir+'/'+name+'/label.txt')  # 打开源文件
    indate = class_source.read()  # 显示所有源文件内容
    if int(indate)==1:
        j+=1
        print(name,",")

    match_txt = dir1 + '/cls_match.txt'
    #print(name)
    open(match_txt, 'a').write(
        "Name: {} >> Class: {}\n".format(name, indate))
print("p:",j)