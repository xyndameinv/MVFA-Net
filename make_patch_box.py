# -*-coding:utf-8-*-
"""
Tools for making train and test patches
"""
import os
import glob
import numpy as np
import nibabel as nib
from skimage import measure
import scipy.ndimage as sni
import scipy

def scaleit(image, factor, isseg=False):
    #缩放
    order = 0 if isseg == True else 3

    height, width, depth= image.shape
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = int(np.round(factor * depth))

    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row     = (height - zheight) // 2
        col     = (width - zwidth) // 2
        layer   = (depth - zdepth) // 2
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = sni.interpolation.zoom(image, (float(factor), float(factor), float(factor)), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        layer   = (zdepth - depth) // 2

        newimg = sni.interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), float(factor)), order=order, mode='nearest')

        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        return newimg

    else:
        return image

def rotateit(image, theta, isseg=False):
    #旋转
    order = 0 if isseg == True else 5
    return sni.rotate(image, float(theta), reshape=False, order=order, mode='nearest')

def normalize_data(data):
    # b = np.percentile(data, 98)
    # t = np.percentile(data, 1)
    # data = np.clip(data,t,b)
    data = np.array(data,dtype=np.float32)
    means = data.mean()
    stds = data.std()
    # print(type(data),type(means),type(stds))
    data -= means
    data /= stds
    return data


def getRangImageDepth(label):  # 3D
    """
    :param image:
    :return:rangofimage depth
    """
    fistflag = True
    startpositionz = 0
    endpositionz = 0
    for z in range(label.shape[2]):
        notzeroflag = np.max(label[:, :, z])
        if notzeroflag and fistflag:
            startpositionz = z
            fistflag = False
        if notzeroflag:
            endpositionz = z
    return startpositionz, endpositionz


def getRangImagex(label):
    """
    :param image:
    :return:rangofimage depth
    """
    fistflagx = True
    startpositionx = 0
    endpositionx = 0
    for x in range(label.shape[0]):
        notzeroflagx = np.max(label[x, :, :])
        if notzeroflagx and fistflagx:
            startpositionx = x
            fistflagx = False
        if notzeroflagx:
            endpositionx = x
    return startpositionx, endpositionx


def getRangImagey(label):
    """
    :param image:
    :return:rangofimage depth
    """
    fistflagy = True
    startpositiony = 0
    endpositiony = 0
    for y in range(label.shape[1]):
        notzeroflagy = np.max(label[:, y, :])
        if notzeroflagy and fistflagy:
            startpositiony = y
            fistflagy = False
        if notzeroflagy:
            endpositiony = y
    return startpositiony, endpositiony


def get_binary_xy(in_file, dim_num=2):
    '''
    获取二维二值矩阵的轮廓xy-plane
    in_file:为输入的矩阵
    return:contour的坐标
    '''
    binary_xy = []
    if dim_num > 2:
        print('the input must be two dims.')
        exit()
    contours = measure.find_contours(in_file, 0.5)
    contour = contours[0]
    contour = np.around(contour, 1).astype(np.int)
    #    contour[:, [0, 1]] = contour[:, [1, 0]]

    for i in range(contour.shape[0]):
        binary_xy.append(list(contour[i, :]))

    return binary_xy


def get_target_range(in_file):
    '''
    获取in_file矩阵里,最大非0边界范围  (目标)
    in_file:为输入的矩阵
    return:为范围的min/max
    '''
    get_binar = np.where(in_file == 1)

    w, h, d = (get_binar[0], get_binar[1], get_binar[2])
    #    max+1是因为在in_file[w_min:w_max,h_min:h_max,d_min:d_max]，
    #   不会取到max,需要+1，确保所有label==1的取完整
    w_min, w_max = (w.min(), w.max() + 1)
    h_min, h_max = (h.min(), h.max() + 1)
    d_min, d_max = (d.min()+1, d.max() + 1)
    return w_min, w_max, h_min, h_max, d_min, d_max


def get_binary_coors(in_file, dim_num=2):
    '''
    获取三维的轮廓坐标
    in_file:为输入的矩阵（label）
    return:binary_coor_list为轮廓坐标[(w0,h0,d0),(w1,h1,d1),...]
            center_contour_img是输入对应的骨架线图矩阵
    '''
    center_contour_img = np.zeros(in_file.shape)
    w_min, w_max, h_min, h_max, d_min, d_max = get_target_range(in_file)
    usege_file = in_file[:, :, d_min:d_max]
    binary_coors = []
    for i in range(usege_file.shape[2]):
        binary_xy = get_binary_xy(usege_file[:, :, i], dim_num=2)
        for j in range(len(binary_xy)):
            binary_xy[j].append(i + d_min)
            binary_coors.extend(binary_xy)

    for i in range(len(binary_coors)):
        xyz = binary_coors[i]
        center_contour_img[xyz[0], xyz[1], xyz[2]] = 1.0

    binary = np.where(center_contour_img)
    binary_coor_list = []
    for k in range(binary[0].shape[0]):
        coor = (binary[0][k], binary[1][k], binary[2][k])
        binary_coor_list.append(coor)

    return binary_coor_list, center_contour_img


# def pad_num(x,y,z,px,py,pz):
#     if px - 16 < 0:
#         padx0 = 32 - px
#     if py - 16 < 0:
#         pady0 = 32 - py
#     if pz - 16 < 0:
#         padz0 = 16 - pz
#     if x - px < 16:
#         padx1 = 16 - (x - px)
#     if y - py < 16:
#         pady1 = 16 - (y - py)
#     if z - pz < 16:
#         padz1 = 16 - (z - pz)
#     return padx0,padx1,pady0,pady1,padz0,padz1

def adjust_xyz(x,y,z,px,py,pz):
    if px - 32 < 0:
        px = px + 32
    if py - 32 < 0:
        py = py + 32
    if pz - 16 < 0:
        pz = pz + 16
    if x - px < 32:
        px = px - 32
    if y - py < 32:
        py = py - 32
    if z - pz < 16:
        pz = pz - 16

    if pz<16 or pz > z-16:
        pz = np.random.randint(16,z-16)
    return px,py,pz

def make_random_patch(img_arr, label_arr, patch_num,patch_size):
    # img = [x,y,z]
    # return  list [64,64,64]
    x,y,z = label_arr.shape
    sub_data_shape = tuple([patch_num] + list(patch_size) + [1])
    sub_gt_shape = tuple([patch_num] + list(patch_size) + [1])
    sub_data = np.ones(shape=sub_data_shape, dtype=np.float32)
    sub_gt = np.ones(shape=sub_gt_shape, dtype=np.uint8)
    # startpointz, endpointz = getRangImageDepth(label_arr)
    # startpointX, endpointX = getRangImagex(label_arr)
    # startpointY, endpointY = getRangImagey(label_arr)
    # boundary_seeds_list,center_contour_img = get_binary_coors(label_arr, dim_num=2)
    # boundary_seed_num = len(boundary_seeds_list)
    patch_z = int(patch_size[2]/2)
    for i in range(patch_num):
        # px = np.random.randint(int(x/4),int(x/4 *3))
        # py = np.random.randint(int(y/4),int(y/4 *3))
        px = np.random.randint(64, x-64)
        py = np.random.randint(64, y-64)
        pz = np.random.randint(4,z-4)
        sub_img = img_arr[px - 64:px + 64, py - 64:py + 64, pz - patch_z:pz + patch_z]
        sub_label = label_arr[px-64:px + 64, py-64:py + 64, pz-patch_z:pz + patch_z]
        sub_data[i, :, :, :, 0] = sub_img
        sub_gt[i, :, :, :, 0] = sub_label
        # if i%5 == 0 or i%5 == 1: #boundary
        #     random_seed = np.random.randint(boundary_seed_num)
        #     px,py,pz = boundary_seeds_list[random_seed]
        #     px,py,pz = adjust_xyz(x,y,z,px,py,pz)
        #     sub_img = img_arr[px-32:px + 32, py-32:py + 32, pz-16:pz + 16]
        #     sub_label = label_arr[px-32:px + 32, py-32:py + 32, pz-16:pz + 16]
        #     sub_data[i, :, :, :, 0] = sub_img
        #     sub_gt[i, :, :, :, 0] = sub_label
        # elif i%5 == 2 or i%5 == 3: #target area
        #     px = np.random.randint(startpointX,endpointX)
        #     py = np.random.randint(startpointY,endpointY)
        #     pz = np.random.randint(startpointz,endpointz)
        #     px,py,pz = adjust_xyz(x,y,z,px,py,pz)
        #     sub_img = img_arr[px-32:px + 32, py-32:py + 32, pz-16:pz + 16]
        #     sub_label = label_arr[px-32:px + 32, py-32:py + 32, pz-16:pz + 16]
        #     sub_data[i, :, :, :, 0] = sub_img
        #     sub_gt[i, :, :, :, 0] = sub_label
        # else:           #non target area
        #     # px = np.random.randint(32+40,x-32-40)
        #     # py = np.random.randint(32+40,y-32-40)
        #     # pz = np.random.randint(16,z-16)
        #     px = np.random.randint(32, x - 32)
        #     py = np.random.randint(32, y - 32)
        #     pz = np.random.randint(16, z - 16)
        #     px, py, pz = adjust_xyz(x, y, z, px, py, pz)
        #     sub_img = img_arr[px-32:px + 32, py-32:py + 32, pz-16:pz + 16]
        #     sub_label = label_arr[px-32:px + 32, py-32:py + 32, pz-16:pz + 16]
        #     sub_data[i, :, :, :, 0] = sub_img
        #     sub_gt[i, :, :, :, 0] = sub_label
    return sub_data,sub_gt




def make_train_patch(src_path, patch_num, patch_size):  # src_path = 'D:\3DMRI\NewObj\data\rename\stomach'
    sto = os.listdir(src_path)
    sub_num = len(sto)
    train_data_shape = tuple([sub_num * patch_num] + list(patch_size) + [1])
    train_label_shape = tuple([sub_num * patch_num] + list(patch_size) + [1])
    # train_data = np.ones(shape=train_data_shape, dtype=np.float32)
    # train_label = np.ones(shape=train_label_shape, dtype=np.uint8)
    j = 0
    for substo in sto:
        print("make patch in {}".format(substo))
        imgs = glob.glob(os.path.join(src_path, substo, '*.nii.gz'))
        if 'LABEL' in imgs[0] or 'LABEL' in imgs[1] or 'LABLE' in imgs[0] or 'LABLE' in imgs[1]:
            img = nib.load(imgs[1]).get_data()
            label = nib.load(imgs[0]).get_data()
        else:
            img = nib.load(imgs[0]).get_data()
            label = nib.load(imgs[1]).get_data()
        img = np.squeeze(img)
        img = normalize_data(img)
        label = np.squeeze(label)
        affine = nib.load(imgs[0]).affine

        # label[label > 0] = 1
        # label = normalize_data(label)
        sub_data, sub_gt = make_random_patch(img, label, patch_num, patch_size)

        #save patches
        patch_img_save_folder = r'..\data\128-8-from_crop_0.50.54\img_128_8'
        patch_label_save_folder = r'..\data\128-8-from_crop_0.50.54\label_128_8'
        if not os.path.exists(patch_img_save_folder):
            os.mkdir(patch_img_save_folder)
        if not os.path.exists(patch_label_save_folder):
            os.mkdir(patch_label_save_folder)
        for m in range(patch_num):
            new_patch = nib.Nifti1Image(sub_data[m,...,0],affine)
            nib.save(new_patch, patch_img_save_folder+'/'+substo+'_'+str(m)+'.nii.gz')

            #随机旋转90,180,270
            # theta_arr = [-15, -10, 10,15]
            # theta = np.random.choice(theta_arr, 1)[0]

            theta = 180
            rota_patch_img_arr = rotateit(sub_data[m,...,0], theta, isseg=False)
            rota_patch_img = nib.Nifti1Image(rota_patch_img_arr,affine)
            nib.save(rota_patch_img,patch_img_save_folder+'/'+substo+'_'+str(m)+'rot'+'.nii.gz')

            # #随机缩放0.7~1.3
            # factor_arr = [0.7,0.8,0.9,1.1,1.2,1.3]
            # factor = np.random.choice(factor_arr,1)[0]
            # sca_patch_img_arr = scaleit(sub_data[m,0,...],factor,isseg=False)
            # sca_patch_img = nib.Nifti1Image(sca_patch_img_arr,affine)
            # nib.save(sca_patch_img,patch_img_save_folder+'/'+substo+'_'+str(m)+'sca'+'.nii.gz')

            #处理label
            new_label_patch = nib.Nifti1Image(sub_gt[m, ..., 0], affine)
            nib.save(new_label_patch, patch_label_save_folder + '/' + substo + '_' + str(m) + '.nii.gz')

            rota_patch_gt_arr = rotateit(sub_gt[m, ..., 0], theta, isseg=True)
            rota_patch_gt = nib.Nifti1Image(rota_patch_gt_arr, affine)
            nib.save(rota_patch_gt, patch_label_save_folder + '/' + substo + '_' + str(m) + 'rot' + '.nii.gz')

            # sca_patch_gt_arr = scaleit(sub_gt[m, 0, ...], factor, isseg=True)
            # sca_patch_gt = nib.Nifti1Image(sca_patch_gt_arr, affine)
            # nib.save(sca_patch_gt, patch_label_save_folder + '/' + substo + '_' + str(m) + 'sca' + '.nii.gz')


        # patch_label_save_folder = r'G:\zxp\3DMRI\NewObj\code_low_gen\label_64'
        # if not os.path.exists(patch_label_save_folder):
        #     os.mkdir(patch_label_save_folder)
        # for n in range(patch_num):
        #     new_label_patch = nib.Nifti1Image(sub_gt[n, 0, ...], affine)
        #     nib.save(new_label_patch, patch_label_save_folder + '/' + substo + '_' + str(n) + '.nii.gz')
        #
        #     # 随机旋转-15~+15
        #     # theta_arr = [-15,-10,10,15]
        #     # theta = np.random.choice(theta_arr,1)[0]
        #     rota_patch_gt_arr = rotateit(sub_gt[n, 0, ...], theta, isseg=True)
        #     rota_patch_gt = nib.Nifti1Image(rota_patch_gt_arr, affine)
        #     nib.save(rota_patch_gt, patch_label_save_folder + '/' + substo + '_' + str(n) + 'rot' + '.nii.gz')
        #
        #     # 随机缩放0.7~1.3
        #     # factor_arr = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
        #     # factor = np.random.choice(factor_arr, 1)[0]
        #     sca_patch_gt_arr = scaleit(sub_gt[n, 0, ...], factor, isseg=True)
        #     sca_patch_gt = nib.Nifti1Image(sca_patch_gt_arr, affine)
        #     nib.save(sca_patch_gt, patch_label_save_folder + '/' + substo + '_' + str(n) + 'sca' + '.nii.gz')

        # train_data[j * patch_num:(j + 1) * patch_num,...] = sub_data
        # train_label[j * patch_num:(j + 1) * patch_num,...] = sub_gt
        j += 1
    #np.save('data_64_2000.npy',train_data)
    #np.save('label_64_2000.npy',train_label)


    # #print tongji xinxi
    # nums_patch = train_data.shape[0]
    # num_one = 0
    # half_voxel = 0
    # for i in range(nums_patch):
    #     if train_label[i].any()==1:
    #         num_one += 1
    #     if np.sum(train_label[i])>10000:
    #         half_voxel += 1
    #
    # print('half voxel',half_voxel)
    # voxel_one = np.sum(train_label)
    # voxel_total = train_label.shape[0]*train_label.shape[1]*train_label.shape[2]*train_label.shape[3]*train_label.shape[4]
    # print('voxel_one',voxel_one)
    # print('total voxel',voxel_total)
    # print('voxel_rate',voxel_one/voxel_total)
    #
    # print('rate', num_one/nums_patch)
    # print("num_one",num_one)
    # print("nums_patch",nums_patch)


    # train_label2 = get_multi_class_labels(train_label, n_labels, labels)
    # return train_data, train_label
def sort_patches(img_folder,label_folder,threshold=10):
    # imgs = glob.glob(img_folder+'/'+'*.nii.gz')
    #目标小的或者没有的块直接删掉
    labels = glob.glob(label_folder + '/'+'*.nii.gz')
    for label in labels:
        label_arr = nib.load(label).get_data()
        label_one = np.sum(label_arr)
        if label_one < threshold:
            os.remove(label)
            img_file = img_folder + '/'+os.path.basename(label)
            os.remove(img_file)

if __name__ == '__main__':
    src_path = r'..\data\lasttrain_0.50.54'
    patch_num = 50
    patch_size = (128,128,8)
    make_train_patch(src_path, patch_num, patch_size)
    print('make patches has done,now start sort: ')
    img_folder = r'..\data\128-8-from_crop_0.50.54\img_128_8'
    label_folder = r'..\data\128-8-from_crop_0.50.53=4\label_128_8'
    sort_patches(img_folder, label_folder,threshold=4)
