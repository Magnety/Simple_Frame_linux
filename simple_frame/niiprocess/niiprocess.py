import SimpleITK as sitk
import numpy
import os


def redo_spacing(img,label,path):
    label.SetOrigin(img.GetOrigin())
    label.SetSpacing(img.GetSpacing())
    sitk.WriteImage(label,path+'/segmentation.nii.gz')



dir = "/home/ubuntu/liuyiyao/Simple_Frame_data/breast_data_153_noclsmask"
names = os.listdir(dir)
for name in names:
    path = dir +'/'+name
    img = sitk.ReadImage(path+'/imaging.nii.gz')
    label = sitk.ReadImage(path+'/segmentation.nii.gz')
    redo_spacing(img,label,path)