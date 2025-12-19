import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import random
import lmdb
import pandas as pd
from monai import transforms
from scipy import ndimage


def downsample_4x_3d(img, factor=4,annotation=False):
    # 获取原始图像的维度
    
    depth, height, width = img.shape

    # 计算新的深度、高度和宽度，使其可以整除 factor
    new_depth = depth // factor * factor
    new_height = height // factor * factor
    new_width = width // factor * factor

    # 裁剪图像以确保每个维度都能被 factor 整除
    img = img[:new_depth, :new_height, :new_width]
    if annotation == True:
        print(f'---------测试代码: 裁剪图像后{img.shape}')

    # 对每 4 层、4 行、4 列进行池化
    # 对深度进行压缩
    # img = img.reshape(new_depth // factor, factor, new_height, new_width)  # 重新排列为 (depth//factor, factor, height, width)
    # img = img.mean(axis=1)  # 对深度轴进行池化，计算每 4 层的平均
    # if annotation == True:
    #     print(f'---------测试代码: 深度轴进行压缩后{img.shape}')

    # 对高度进行压缩
    # img = img.reshape(new_depth // factor, new_height // factor, factor, new_width)  # 重新排列为 (depth, height//factor, factor, width)
    # img = img.mean(axis=2)  # 对高度轴进行池化，计算每 4 行的平均
    # if annotation == True:
    #     print(f'---------测试代码: 高度轴进行压缩后{img.shape}')

#     # 对宽度进行压缩
#     img = img.reshape(new_depth // factor, new_height // factor, new_width // factor, factor)  # 重新排列为 (depth, height, width//factor, factor)
#     img = img.mean(axis=3)  # 对宽度轴进行池化，计算每 4 列的平均
#     if annotation == True:
#         print(f'---------测试代码: 宽度轴进行压缩后{img.shape}')


    # 对宽度进行压缩
    img = img.reshape(new_depth, new_height, new_width // factor, factor)  # 重新排列为 (depth, height, width//factor, factor)
    img = img.mean(axis=3)  # 对宽度轴进行池化，计算每 4 列的平均
    if annotation == True:
        print(f'---------测试代码: 宽度轴进行压缩后{img.shape}')
    # 对高度进行压缩
#     img = img.reshape(new_depth, new_height // factor, factor, new_width)  # 重新排列为 (depth, height//factor, factor, width)
#     img = img.mean(axis=2)  # 对高度轴进行池化，计算每 4 行的平均
#     if annotation == True:
#         print(f'---------测试代码: 高度轴进行压缩后{img.shape}')


    return img

class MyDataseGM(Dataset):

    def __init__(self, data_list_path=None, env=None, train=False,datatype=None,Data=None, shape=[128,128,128]):
        # self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        # 有表头用下面的
        self.Type = datatype
        if self.Type == None:
            self.data_list = np.genfromtxt(data_list_path, dtype=str, delimiter=',', skip_header=1)
        else:
            self.data = Data
        self.env = env
        self.train = train
        self.aug_mode = random.randint(0, 12)
        self.shape = shape
        nifti_image = nib.load("/home/caojiaxiang/brain age/Third/ROI_MNI_V4.nii")
        self.aaldata = np.array(nifti_image.get_fdata())
        self.affine = nifti_image.affine

    def __len__(self):  # 返回整个数据集的大小
        if self.Type == None:
            return len(self.data_list)
        else:
            return len(self.data)
        

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        if self.Type == None:
            dataset_id, dataset, site_name, site_id, pid, age, gender = self.data_list[index]
        else:
            pid, age= self.data.iloc[index]
        # pid, age = self.data_list[index]
        



        with self.env.begin(write=False) as txn:
#             buf = txn.get(pid.encode())
            buf = txn.get(f"{pid}_gm".encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)
#         img = (x-np.min(x))/(np.max(x)-np.min(x))
        img = x
        '''translate 10'''
        # if self.train:
        #     coordinateTransformWrapper(img, maxDeg=0, maxShift=10, mirror_prob=0.)

        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)



        '''3d，数据增强  random value '''
        if self.train:
            img = self.apply_augment(img)
        else:
            resize_op = transforms.Resize(self.shape)
            img = resize_op(img)
        
        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample
    def apply_augment(self, img):
        if self.aug_mode == 1:
            x, y, z = gen_cord(imgsize=(91, 109, 91))
            crop_op = transforms.CenterSpatialCrop([x, y, z])
            img = crop_op(img)
        elif self.aug_mode == 2:
            rotate_op = transforms.Rotate(angle=[random.randint(0, 45),random.randint(0, 45),random.randint(0, 45)])
            img = rotate_op(img)
        elif self.aug_mode == 3:
            flip_op = transforms.Flip(spatial_axis=None)
            img = flip_op(img)
        elif self.aug_mode == 4:
            flip_op = transforms.RandFlip(prob=0.5, spatial_axis=random.randint(0, 2))
            img = flip_op(img)
        elif self.aug_mode == 5:
            flip_op = transforms.RandAxisFlip(prob=0.5)
            img = flip_op(img)
        elif self.aug_mode == 6:
            gaussian_op = transforms.RandGaussianNoise(std=0.02)
            img = gaussian_op(img)
        elif self.aug_mode == 7:
            gaussian_op = transforms.RandGaussianNoise(std=0.1)
            img = gaussian_op(img)
        elif self.aug_mode == 8:
            x, y, z = gen_cord(imgsize=(91, 109, 91))
            crop_op = transforms.CenterSpatialCrop([x, y, z])
            img = crop_op(img)
            flip_op = transforms.RandFlip(prob=0.5, spatial_axis=random.randint(0, 2))
            img = flip_op(img)
        resize_op = transforms.Resize(self.shape)
        return resize_op(img)
    

class MyDataseWM(MyDataseGM):
    def __getitem__(self, index):
        if self.Type is None:
            dataset_id, dataset, site_name, site_id, pid, age, gender = self.data_list[index]
        else:
            pid, age = self.data.iloc[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(f"{pid}_wm".encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)
        img = (x - np.min(x)) / (np.max(x) - np.min(x))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        if self.train:
            img = self.apply_augment(img)
        else:
            resize_op = transforms.Resize(self.shape)
            img = resize_op(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label
        return sample
    

class MyDataseCSF(MyDataseGM):
    def __getitem__(self, index):
        if self.Type is None:
            dataset_id, dataset, site_name, site_id, pid, age, gender = self.data_list[index]
        else:
            pid, age = self.data.iloc[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(f"{pid}_csf".encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)
        img = (x - np.min(x)) / (np.max(x) - np.min(x))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        if self.train:
            img = self.apply_augment(img)
        else:
            resize_op = transforms.Resize(self.shape)
            img = resize_op(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label
        return sample
    

class MyDataseGmWmCsf(MyDataseGM):
    def __getitem__(self, index):
        if self.Type is None:
            dataset_id, dataset, site_name, site_id, pid, age, gender = self.data_list[index]
        else:
            pid, age = self.data.iloc[index]

        with self.env.begin(write=False) as txn:
            buf_gm = txn.get(f"{pid}_gm".encode())
            buf_wm = txn.get(f"{pid}_wm".encode())
            buf_csf = txn.get(f"{pid}_csf".encode())

        gm = np.frombuffer(buf_gm, dtype=np.float32).reshape(91, 109, 91)
        wm = np.frombuffer(buf_wm, dtype=np.float32).reshape(91, 109, 91)
        csf = np.frombuffer(buf_csf, dtype=np.float32).reshape(91, 109, 91)

        # Normalize each modality separately
        def norm(img):
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        gm = norm(gm)
        wm = norm(wm)
        csf = norm(csf)

        img = np.stack([gm, wm, csf], axis=0)  # Shape: [3, 91, 109, 91]
        img = torch.from_numpy(img)

#         if self.train:
#             img = self.apply_augment(img)
#         else:
#             resize_op = transforms.Resize([3, 91, 91, 109])
#             img = resize_op(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label
        return sample
    

class MyDataseGmWmCsfT1w(MyDataseGM):
    def __init__(self, data_list_path=None, env=None, train=False, datatype=None, Data=None, shape=[128,128,128], env2=None):
        super().__init__(data_list_path=data_list_path, env=env, train=train, datatype=datatype, Data=Data, shape=shape)
        self.env2 = env2
    def __getitem__(self, index):
        if self.Type is None:
            dataset_id, dataset, site_name, site_id, pid, age, gender = self.data_list[index]
        else:
            pid, age = self.data.iloc[index]

        with self.env.begin(write=False) as txn:
            buf_gm = txn.get(f"{pid}_gm".encode())
            buf_wm = txn.get(f"{pid}_wm".encode())
            buf_csf = txn.get(f"{pid}_csf".encode())
        with self.env2.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        gm = np.frombuffer(buf_gm, dtype=np.float32).reshape(91, 109, 91)
        wm = np.frombuffer(buf_wm, dtype=np.float32).reshape(91, 109, 91)
        csf = np.frombuffer(buf_csf, dtype=np.float32).reshape(91, 109, 91)
        t1 = np.frombuffer(buf, dtype=np.float32).reshape(91, 109, 91)

        # Normalize each modality separately
        def norm(img):
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        gm = norm(gm)
        wm = norm(wm)
        csf = norm(csf)
        t1 = norm(t1)

        img = np.stack([gm, wm, csf,t1], axis=0)  # Shape: [3, 91, 109, 91]
        img = torch.from_numpy(img)

#         if self.train:
#             img = self.apply_augment(img)
#         else:
#             resize_op = transforms.Resize([3, 91, 91, 109])
#             img = resize_op(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label
        return sample
    
    
class MyDataseGmWm(MyDataseGM):
    def __getitem__(self, index):
        if self.Type is None:
            dataset_id, dataset, site_name, site_id, pid, age, gender = self.data_list[index]
        else:
            pid, age = self.data.iloc[index]

        with self.env.begin(write=False) as txn:
            buf_gm = txn.get(f"{pid}_gm".encode())
            buf_wm = txn.get(f"{pid}_wm".encode())


        gm = np.frombuffer(buf_gm, dtype=np.float32).reshape(91, 109, 91)
        wm = np.frombuffer(buf_wm, dtype=np.float32).reshape(91, 109, 91)

        # Normalize each modality separately
        def norm(img):
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        gm = norm(gm)
        wm = norm(wm)


        img = np.stack([gm, wm], axis=0)  # Shape: [2, 91, 109, 91]
        img = torch.from_numpy(img)

#         if self.train:
#             img = self.apply_augment(img)
#         else:
#             resize_op = transforms.Resize([3, 91, 91, 109])
#             img = resize_op(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label
        return sample
    
    
# class MyDataseT1w(MyDataseGM):
#     def __getitem__(self, index):
#         if self.Type is None:
#             pid, age= self.data_list[index]
#         else:
#             pid, age = self.data.iloc[index]

#         with self.env.begin(write=False) as txn:
#             buf = txn.get(pid.encode())
#         img_flat = np.frombuffer(buf, dtype=np.float32)
#         x = img_flat.copy().reshape(91, 109, 91)
#         img = (x - np.min(x)) / (np.max(x) - np.min(x))
#         img = np.expand_dims(img, 0)
#         img = torch.from_numpy(img)

#         if self.train:
#             img = self.apply_augment(img)
#         else:
#             resize_op = transforms.Resize(self.shape)
#             img = resize_op(img)

#         label = torch.FloatTensor([float(age)])
#         sample = img, label
#         return sample
    
    
class MyDataseT1w(Dataset):

    def __init__(self, data_list_path=None, env=None, train=False,datatype=None,Data=None, shape=[128,128,128]):
        # self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        # 有表头用下面的
        self.Type = datatype
        if self.Type == None:
            self.data_list = np.genfromtxt(data_list_path, dtype=str, delimiter=',', skip_header=1)
        else:
            self.data = Data
        self.env = env
        self.train = train
        self.shape = shape
    
        # 被注释的两个变换的计算量太大
        self.train_transforms = transforms.Compose([
            transforms.RandSpatialCrop(roi_size=(80, 96, 80), random_center=True, random_size=False),
            transforms.Resize(spatial_size=shape),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            transforms.RandAffine(prob=0.3, translate_range=(5,5,5), scale_range=(0.05,0.05,0.05)),
            transforms.RandGaussianNoise(prob=0.3, std=0.05),
            transforms.RandBiasField(prob=0.3),
            transforms.RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(spatial_size=shape),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])

    def __len__(self):  # 返回整个数据集的大小
        if self.Type == None:
            return len(self.data_list)
        else:
            return len(self.data)
        

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        if self.Type == None:
            pid, age= self.data_list[index]
        else:
            pid, age= self.data.iloc[index]
        # pid, age = self.data_list[index]
        



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)



        img = np.expand_dims(x, 0)
        img = torch.from_numpy(img)



        '''3d，数据增强  random value '''
        if self.train:
            img = self.train_transforms(img)
            
        else:
            img = self.test_transforms(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample




class MyDataseT1wRBA(Dataset):

    def __init__(self, data_list_path=None, env=None, train=False,datatype=None,Data=None, shape=[128,128,128]):
        # self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        # 有表头用下面的
        self.Type = datatype
        if self.Type == None:
            self.data_list = np.genfromtxt(data_list_path, dtype=str, delimiter=',', skip_header=1)
        else:
            self.data = Data
        self.env = env
        self.train = train
        self.shape = shape
        self.region_tensor = torch.load("/home/caojiaxiang/brain age/RBA/pths/AAL_regions_116.pt")  # [116, 91, 109, 91]



    
        # 被注释的两个变换的计算量太大
        self.train_transforms = transforms.Compose([
            # transforms.RandSpatialCrop(roi_size=(80, 96, 80), random_center=True, random_size=False),
            transforms.Resize(spatial_size=shape),
            # transforms.RandFlip(prob=0.5, spatial_axis=0),
            # transforms.RandFlip(prob=0.5, spatial_axis=1),
            # transforms.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            transforms.RandAffine(prob=0.2, translate_range=(5,5,5), scale_range=(0.05,0.05,0.05)),
            transforms.RandGaussianNoise(prob=0.2, std=0.05),
            transforms.RandBiasField(prob=0.2),
            transforms.RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(spatial_size=shape),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])

    def __len__(self):  # 返回整个数据集的大小
        if self.Type == None:
            return len(self.data_list)
        else:
            return len(self.data)
        

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        if self.Type is None:
            pid, age= self.data_list[index]
        else:
            pid, age= self.data.iloc[index]
        # pid, age = self.data_list[index]
        



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)



        img = np.expand_dims(x, axis=0)
        img = torch.from_numpy(img)



        '''3d，数据增强  random value '''
        if self.train:
            img = self.train_transforms(img)
            
        else:
            img = self.test_transforms(img)

        label = torch.FloatTensor([float(age)])

        # 创建全零张量
        rba_age = torch.zeros(117, dtype=torch.float32)

        # 将 age 复制到第1~116列
        rba_age[1:] = label.repeat(116)

        rand_region_idx = torch.randint(0, 117, (1,)).item()
        region_mask = self.region_tensor[rand_region_idx, :, :, :].float()  # [D, H, W]
        noise = torch.randn_like(region_mask) * 0.3 * region_mask
        noise_img = img.clone()

        noise_img = noise_img + noise.unsqueeze(0)
        # print(noise_img.shape)
        sample = img, label, self.region_tensor,rba_age,noise_img,rand_region_idx

        return sample


class MyDataseT1w_iD(Dataset):

    def __init__(self, data_list_path=None, env=None, train=False,datatype=None,Data=None, shape=[128,128,128]):
        # self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        # 有表头用下面的
        self.Type = datatype
        if self.Type == None:
            self.data_list = np.genfromtxt(data_list_path, dtype=str, delimiter=',', skip_header=1)
        else:
            self.data = Data
        self.env = env
        self.train = train
        self.shape = shape
    
        # 被注释的两个变换的计算量太大
        self.train_transforms = transforms.Compose([
            transforms.RandSpatialCrop(roi_size=(80, 96, 80), random_center=True, random_size=False),
            transforms.Resize(spatial_size=shape),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            transforms.RandAffine(prob=0.3, translate_range=(5,5,5), scale_range=(0.05,0.05,0.05)),
            transforms.RandGaussianNoise(prob=0.3, std=0.05),
            transforms.RandBiasField(prob=0.3),
            transforms.RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(spatial_size=shape),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])

    def __len__(self):  # 返回整个数据集的大小
        if self.Type == None:
            return len(self.data_list)
        else:
            return len(self.data)
        

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        if self.Type == None:
            pid, age= self.data_list[index]
        else:
            pid, age= self.data.iloc[index]
        # pid, age = self.data_list[index]
        



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)



        img = np.expand_dims(x, 0)
        img = torch.from_numpy(img)



        '''3d，数据增强  random value '''
        if self.train:
            img = self.train_transforms(img)
            
        else:
            img = self.test_transforms(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label,pid

        return sample


class MyDataseT1w_4to1(Dataset):

    def __init__(self, data_list_path=None, env=None, train=False,datatype=None,Data=None, shape=[128,128,128]):
        # self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        # 有表头用下面的
        self.Type = datatype
        if self.Type == None:
            self.data_list = np.genfromtxt(data_list_path, dtype=str, delimiter=',', skip_header=1)
        else:
            self.data = Data
        self.env = env
        self.train = train
        self.shape = shape
    
        # 被注释的两个变换的计算量太大
        self.train_transforms = transforms.Compose([
            transforms.RandSpatialCrop(roi_size=(80, 96, 80), random_center=True, random_size=False),
            transforms.Resize(spatial_size=shape),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            transforms.RandAffine(prob=0.3, translate_range=(5,5,5), scale_range=(0.05,0.05,0.05)),
            transforms.RandGaussianNoise(prob=0.3, std=0.05),
            transforms.RandBiasField(prob=0.3),
            transforms.RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(spatial_size=shape),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
        ])

    def __len__(self):  # 返回整个数据集的大小
        if self.Type == None:
            return len(self.data_list)
        else:
            return len(self.data)
        

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        if self.Type == None:
            pid, age= self.data_list[index]
        else:
            pid, age= self.data.iloc[index]
        # pid, age = self.data_list[index]
        



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)
        x = downsample_4x_3d(x, factor=4)



        img = np.expand_dims(x, 0)
        img = torch.from_numpy(img)



        '''3d，数据增强  random value '''
        if self.train:
            img = self.train_transforms(img)
            
        else:
            img = self.test_transforms(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample



import os
import time
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import lmdb
from scipy.sparse import csr_matrix
from scipy.ndimage import zoom
from monai.transforms import Compose, EnsureChannelFirstd, RandFlipd, RandGaussianNoised, EnsureTyped

import pickle
import hashlib



class HybridROIDataset(Dataset):
    def __init__(self, 
                 data_list_path: str,
                 env: lmdb.Environment,
                 aal_path: str = "/home/caojiaxiang/brain age/Third/ROI_MNI_V4.nii",
                 roi_level: int = 2,
                 zoom_factor: float = 1.3,
                 train: bool = False,
                 target_shape: Tuple[int, int, int] = (64, 64, 64)):
        """
        纯CPU优化的脑区数据集
        
        主要特点：
        1. 完全在CPU上运行，无需GPU
        2. 优化的内存管理和处理流程
        3. 高效的ROI提取和缩放
        4. 预计算掩码和边界框加速处理
        
        Args:
            roi_level: 1=按首位数字分组, 2=按前两位数字分组, 3=单独ROI
            zoom_factor: 脑区放大比例 (1.0为不放大)
        """
        # 初始化配置
        self.env = env
        self.train = train
        self.target_shape = target_shape
        self.zoom_factor = zoom_factor
        
        # 检查AAL模板文件是否存在
        if not os.path.exists(aal_path):
            raise FileNotFoundError(f"AAL模板文件未找到: {aal_path}")
        
        # 加载本地AAL模板
        self.aal_img = nib.load(aal_path)
        self.aal_data = np.asarray(self.aal_img.dataobj)

        self.roi_groups = self._build_roi_groups(roi_level)
        
        # 尝试加载预计算的掩码和边界框
        self.roi_bboxes, self.roi_masks = self._load_or_compute_masks_and_bboxes(roi_level, aal_path)

        
        # 构建ROI系统
        # self.roi_groups = self._build_roi_groups(roi_level)
        # self.roi_bboxes, self.roi_masks = self._precompute_masks_and_bboxes()
        self.num_groups = len(self.roi_groups)
        
        # 数据增强管道
        self.transforms = self._build_transforms()
        
        # 加载样本列表
        self.samples = np.loadtxt(data_list_path, dtype=str, delimiter=',', skiprows=1)
        
        # 预计算缩放参数
        self._precompute_scaling_params()

    def _build_roi_groups(self, level: int) -> Dict[str, List[int]]:
        """构建ROI分组字典"""
        unique_rois = [r for r in np.unique(self.aal_data) if r > 0]
        groups = {}
        
        if level == 1:  # 按首位数字
            for roi in unique_rois:
                key = str(int(roi))[0] + 'xxx'
                groups.setdefault(key, []).append(roi)
        elif level == 2:  # 按前两位数字
            for roi in unique_rois:
                key = str(int(roi))[:2] + 'xx'
                groups.setdefault(key, []).append(roi)
        else:  # 单独ROI
            groups = {str(int(roi)): [roi] for roi in unique_rois}
            
        return groups

    def _get_cache_filename(self, roi_level: int, aal_path: str) -> str:
        """生成唯一的缓存文件名"""
        # 使用文件内容和ROI level创建唯一哈希
        file_hash = self._get_file_hash(aal_path)
        basename = os.path.basename(aal_path).split('.')[0]
        return f"roi_cache_{basename}_level{roi_level}_{file_hash}.pkl"

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_or_compute_masks_and_bboxes(self, roi_level: int, aal_path: str) -> Tuple[Dict[str, Tuple], Dict[str, np.ndarray]]:
        """加载或计算ROI掩码和边界框"""
        cache_dir = "./roidata_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, self._get_cache_filename(roi_level, aal_path))
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    print(f"Loading precomputed ROI data from cache: {cache_file}")
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache loading failed ({e}), recomputing masks and bboxes")
        
        # 没有缓存或加载失败，进行计算
        print("Precomputing masks and bboxes (this may take a while)...")
        result = self._precompute_masks_and_bboxes()
        
        # 保存结果到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved precomputed ROI data to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save cache ({e}), results not persisted")
        
        return result

    def _precompute_masks_and_bboxes(self) -> Tuple[Dict[str, Tuple], Dict[str, np.ndarray]]:
        """预计算边界框和掩码 (实际计算逻辑)"""
        bboxes = {}
        masks = {}
        
        for name, rois in self.roi_groups.items():
            # 创建组合掩码 - 这是计算密集部分
            mask = np.isin(self.aal_data, rois)
            masks[name] = mask
            
            # 计算边界框
            coords = np.argwhere(mask)
            if len(coords) > 0:
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                bboxes[name] = (min_coords.tolist(), max_coords.tolist())
            else:
                bboxes[name] = ([0, 0, 0], [0, 0, 0])
                
        return bboxes, masks


    def _precompute_scaling_params(self):
        """预计算缩放参数"""
        # 计算每个ROI的目标尺寸
        self.roi_target_sizes = {}
        for name, (minc, maxc) in self.roi_bboxes.items():
            roi_size = [maxc[i] - minc[i] + 1 for i in range(3)]
            # 如果放大因子有效，则计算缩放后尺寸
            if self.zoom_factor > 1.0:
                scaled_size = [int(s * self.zoom_factor) for s in roi_size]
                # 确保缩放后尺寸不超过目标形状
                scaled_size = [min(s, self.target_shape[i]) for i, s in enumerate(scaled_size)]
                self.roi_target_sizes[name] = scaled_size
            else:
                self.roi_target_sizes[name] = roi_size

    def _build_transforms(self):
        """构建数据增强管道"""
        transforms = [
            EnsureTyped(keys=["img"], data_type="numpy", dtype=np.float32),
            EnsureChannelFirstd(keys=["img"], channel_dim=0)
        ]
        
        if self.train:
            transforms += [
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
                RandGaussianNoised(keys=["img"], std=0.05, prob=0.3)
            ]
        
        return Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_time = time.time()
        pid, age = self.samples[idx]
        
        # 从LMDB加载原始数据
        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img = np.frombuffer(buf, dtype=np.float32).reshape(91, 109, 91)
        load_time = time.time() - start_time
        
        # 处理所有ROI
        roi_start = time.time()
        roi_features = []
        for name in self.roi_groups.keys():
            minc, maxc = self.roi_bboxes[name]
            roi_feature = self._extract_and_scale_roi(img, name, minc, maxc)
            roi_features.append(roi_feature)
        
        # 堆叠所有ROI特征
        features = np.stack(roi_features, axis=0)
        roi_time = time.time() - roi_start
        
        # 应用增强
        sample = {"img": features, "label": float(age)}
        sample = self.transforms(sample)
        
        # 计时信息
        total_time = time.time() - start_time
        # if idx % 100 == 0:
        #     print(f"Sample {idx}: Load={load_time:.4f}s, "
        #           f"ROI={roi_time:.4f}s, "
        #           f"Total={total_time:.4f}s")
        
        return sample["img"], np.array([sample["label"]], dtype=np.float32)

    def _extract_and_scale_roi(self, 
                              img: np.ndarray, 
                              name: str, 
                              minc: list, 
                              maxc: list) -> np.ndarray:
        """
        提取ROI区域并直接插值到目标形状
        返回形状为 [target_shape[0], target_shape[1], target_shape[2]] 的数组
        """
        # 提取ROI区域
        roi_slice = (
            slice(minc[0], maxc[0]+1),
            slice(minc[1], maxc[1]+1),
            slice(minc[2], maxc[2]+1)
        )
        
        # 获取ROI区域并应用掩码
        roi_region = img[roi_slice]
        mask_patch = self.roi_masks[name][roi_slice]
        masked_roi = roi_region * mask_patch
        
        # 如果ROI区域太小或为空，直接返回零数组
        if masked_roi.size == 0 or np.all(masked_roi == 0):
            return np.zeros(self.target_shape, dtype=np.float32)
        
        # 计算缩放因子
        if self.zoom_factor > 1.0:
            # 先放大到缩放尺寸
            scaled_roi = zoom(masked_roi, self.zoom_factor, order=1)
            
            # 如果缩放后尺寸大于目标形状，裁剪到目标形状
            if scaled_roi.shape[0] > self.target_shape[0] or \
               scaled_roi.shape[1] > self.target_shape[1] or \
               scaled_roi.shape[2] > self.target_shape[2]:
                
                # 计算裁剪范围
                start = [
                    (scaled_roi.shape[i] - self.target_shape[i]) // 2
                    for i in range(3)
                ]
                end = [start[i] + self.target_shape[i] for i in range(3)]
                
                # 确保不越界
                start = [max(0, s) for s in start]
                end = [min(scaled_roi.shape[i], end[i]) for i in range(3)]
                
                # 裁剪到目标形状
                scaled_roi = scaled_roi[
                    start[0]:end[0],
                    start[1]:end[1],
                    start[2]:end[2]
                ]
            
            # 如果缩放后尺寸小于目标形状，填充到目标形状
            if scaled_roi.shape[0] < self.target_shape[0] or \
               scaled_roi.shape[1] < self.target_shape[1] or \
               scaled_roi.shape[2] < self.target_shape[2]:
                
                # 计算填充范围
                pad_width = [
                    ((self.target_shape[i] - scaled_roi.shape[i]) // 2,
                     (self.target_shape[i] - scaled_roi.shape[i] + 1) // 2)
                    for i in range(3)
                ]
                
                # 填充到目标形状
                scaled_roi = np.pad(scaled_roi, pad_width, mode='constant')
            
            return scaled_roi
        else:
            # 如果不需要放大，直接缩放或填充到目标形状
            if masked_roi.shape != self.target_shape:
                # 使用插值缩放
                zoom_factors = [
                    self.target_shape[i] / masked_roi.shape[i]
                    for i in range(3)
                ]
                return zoom(masked_roi, zoom_factors, order=1)
            else:
                return masked_roi
