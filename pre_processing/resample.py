#-*-coding:utf-8-*-
import SimpleITK as sitk
import numpy as np
import os
import glob


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2

def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkBSpline, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing) #除法运算
    # new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16) #np.multiply实现对应元素相乘  np.ceil计算大于等于该数的最小整数
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)

def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkBSpline, transform=None,
                           output_pixel_type=None):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)

def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float32).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image
if __name__=='__main__':
    data_path = r'..\..\data\resample1'
    save_path = r'..\..\data\resample3'
    sto = os.listdir(data_path)
    for subfolder in sto:
        print(subfolder)
        files = glob.glob(os.path.join(data_path,subfolder,'*.nii.gz'))
        nums = len(files)
        if files[0].find('LABEL')>0 or files[1].find('LABEL')>0 or files[0].find('LABLE')>0 or files[1].find('LABLE')>0:
            print("FIND LABLE")
            for i in range(nums):
                file = files[i]
                
                img = sitk.ReadImage(file)
                if i==1:
                    print(file)
                    out = sitk_resample_to_spacing(img, new_spacing=(0.5, 0.5, 3), interpolator=sitk.sitkNearestNeighbor, default_value=0.)
                else:
                    out = sitk_resample_to_spacing(img, new_spacing=(0.5, 0.5, 3), interpolator=sitk.sitkBSpline, default_value=0.)
                out = sitk.Abs(out)
                save_folder = os.path.join(save_path, subfolder)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                sitk.WriteImage(out,save_folder +'/' + os.path.basename(file))
        else:
            for i in range(nums):
                file = files[i]
                
                img = sitk.ReadImage(file)
                if i==1:
                    print(file)
                    out = sitk_resample_to_spacing(img, new_spacing=(0.5, 0.5, 3), interpolator=sitk.sitkNearestNeighbor, default_value=0.)
                else:
                    out = sitk_resample_to_spacing(img, new_spacing=(0.5, 0.5, 3), interpolator=sitk.sitkBSpline, default_value=0.)
                out = sitk.Abs(out)
                save_folder = os.path.join(save_path, subfolder)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                sitk.WriteImage(out,save_folder +'/' + os.path.basename(file))

