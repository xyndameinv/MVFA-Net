#-*-coding:utf-8-*-

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
import glob

def scale(file,d_type):
    # 将图像的灰度值归一化到0-255
    image = sitk.ReadImage(file)
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = resacleFilter.Execute(image)
    image = sitk.Cast(image, d_type)
    return image
    # sitk.WriteImage(image, '***.nii.gz')
def convert_1(label):
    label = nib.load(label)
    label_arr = label.get_data()
    affine = label.affine
    # label_arr = label_arr/255
    label_arr = np.array(label_arr,dtype=np.uint8)
    new_label = nib.Nifti1Image(label_arr,affine)
    return new_label

if __name__=='__main__':
    base_folder = r'..\..\data\resample3'
    folders = glob.glob(base_folder+'/'+'*')
    for folder in folders:
        file = glob.glob(folder+'/'+'*.nii.gz')
        out = scale(file[0],sitk.sitkFloat32)
        save_folder = str(file[0])
        # print(save_folder)
        sitk.WriteImage(out,save_folder)
        print(file[0])

        #label = scale(file[0],sitk.sitkUInt8)
        #save_label_folder = str(file[0])
        #sitk.WriteImage(label,save_label_folder)
        #label = convert_1(file[1])
        #save_folder = str(file[1])
        #nib.save(label,save_folder)