import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import nibabel as nib
import os
import csv
import glob
import pandas as pd

def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
 
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
 
    return quality


def binary_dice3d(s,g):
    #dice score of two 3D volumes
    num=np.sum(np.multiply(s, g))
    denom=s.sum() + g.sum() 
    if denom==0:
        return 1
    else:
        return  2.0*num/denom


def sensitivity (seg,ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg ))
    denom=np.sum(ground)
    if denom==0:
        return 1
    else:
        return num/denom

def specificity (seg,ground): 
    #computes false positive rate
    num=np.sum(np.multiply(ground==0, seg ==0))
    denom=np.sum(ground==0)
    if denom==0:
        return 1
    else:
        return  num/denom

def ppv(pred,truth):
    inter = np.sum(np.multiply(pred,truth))
    Sr = np.sum(pred)
    return inter/Sr


def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0,0], order=0)
    east = ndimage.shift(binary_map, [1, 0,0], order=0)
    north = ndimage.shift(binary_map, [0, 1,0], order=0)
    south = ndimage.shift(binary_map, [0, -1,0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border


def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

def Hausdorff_distance(ref,seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref,seg)
    hausdorff_distance = np.max(
        [np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance



def DSC_whole(pred, orig_label):
    #computes dice for the whole tumor
    return binary_dice3d(pred>0,orig_label>0)


def DSC_en(pred, orig_label):
    #computes dice for enhancing region
    return binary_dice3d(pred==4,orig_label==4)


def DSC_core(pred, orig_label):
    #computes dice for core region
    seg_=np.copy(pred)
    ground_=np.copy(orig_label)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return binary_dice3d(seg_>0,ground_>0)



def sensitivity_whole (seg,ground):
    return sensitivity(seg>0,ground>0)

def sensitivity_en (seg,ground):
    return sensitivity(seg==4,ground==4)

def sensitivity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return sensitivity(seg_>0,ground_>0)



def specificity_whole (seg,ground):
    return specificity(seg>0,ground>0)

def specificity_en (seg,ground):
    return specificity(seg==4,ground==4)

def specificity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return specificity(seg_>0,ground_>0)
    

def hausdorff_whole (seg,ground):
    return Hausdorff_distance(seg==0,ground==0)

def hausdorff_en (seg,ground):
    return Hausdorff_distance(seg!=4,ground!=4)

def hausdorff_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return Hausdorff_distance(seg_==0,ground_==0)

def guo_and_qian_seg(pred,truth):
    inter = np.sum(np.multiply(pred,truth))
    Rs = np.sum(truth)
    Os = np.sum(pred)-inter
    Us = np.sum(truth) - inter
    # OR = np.divide(Os,np.sum(Rs,Os))
    # UR = np.divide(Us,np.sum(Rs,Os))
    OR = Os/(Rs+Os)
    UR = Us/(Rs+Os)
    return OR,UR

def main(path):
    header = ("Dice", 'Sen', 'Spe', 'Haus','OR(guo)','UR(qian)','ppv')
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob(path+'/'+'*'):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        truth = np.squeeze(truth)
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        pred = np.squeeze(prediction)
        rows.append([DSC_whole(pred,truth),sensitivity_whole(pred,truth),
                     specificity_whole(pred,truth),hausdorff_whole(pred,truth),
                     guo_and_qian_seg(pred,truth)[0],guo_and_qian_seg(pred,truth)[1],
                     ppv(pred,truth)])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv(path+'/'+'martics.csv')


if __name__=='__main__':
    path = r'.\prediction_3dunet_0.50.53_1'
    main(path)




