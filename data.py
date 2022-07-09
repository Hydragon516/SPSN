import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from fast_slic import Slic
import cv2
import config


def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, depth


def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomRotation(image,label,depth):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    
    return image,label,depth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5,15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0,20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0,30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        
        return im
    
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])

    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0]-1)  
        randY = random.randint(0, img.shape[1]-1)  

        if random.randint(0,1) == 0:  
            img[randX, randY] = 0  
        else:  
            img[randX, randY] = 255 

    return Image.fromarray(img)  


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        
        self.filter_files()
        self.size = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])

        depth = Image.fromarray(255 - np.array(depth))
        
        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        
        image = colorEnhance(image)

        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)
        
        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR) / 255

        np_depth = np.array(depth)
        np_depth = cv2.resize(np_depth, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)
        np_depth = cv2.cvtColor(np_depth, cv2.COLOR_GRAY2BGR)

        slic = Slic(num_components=config.TRAIN['num_superpixels'], compactness=10)
        SS_map = slic.iterate(np_img)
        depth_SS_map = slic.iterate(np_depth)
        
        ###
        SS_map = SS_map + 1
        
        SS_maps = []
        SS_maps_label = []

        for i in range(1, config.TRAIN['num_superpixels'] + 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    SS_maps_label.append(1)
                else:
                    SS_maps_label.append(0)
            else:
                SS_maps_label.append(0)

            SS_maps.append(buffer)
        
        ss_map = np.array(SS_maps)
        ss_maps_label = np.array(SS_maps_label)

        ###
        depth_SS_map = depth_SS_map + 1

        depth_SS_maps = []
        depth_SS_maps_label = []

        for i in range(1, config.TRAIN['num_superpixels'] + 1):
            buffer = np.copy(depth_SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    depth_SS_maps_label.append(1)
                else:
                    depth_SS_maps_label.append(0)
            else:
                depth_SS_maps_label.append(0)

            depth_SS_maps.append(buffer)
        
        depth_ss_map = np.array(depth_SS_maps)
        depth_ss_maps_label = np.array(depth_SS_maps_label)

        ###
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        
        return image, gt, depth, ss_map, ss_maps_label, depth_ss_map, depth_ss_maps_label

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        
        images = []
        gts = []
        depths = []
        
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        
        w, h = img.size
        
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):

    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader


class SalObjDataset_test(data.Dataset):
    def __init__(self, val_image_root, valid_list, testsize):
        self.testsize = testsize
        
        self.images = []
        self.gts = []
        self.depths = []
        
        for valid_name in valid_list:
            image_root = os.path.join(val_image_root, valid_name, "RGB") + "/"
            gt_root = os.path.join(val_image_root, valid_name, "GT") + "/"
            depth_root = os.path.join(val_image_root, valid_name, "depth") + "/"

            new_images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
            new_gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            new_depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')]

            for i in range(len(new_images)):
                self.images.append(new_images[i])
                self.gts.append(new_gts[i])
                self.depths.append(new_depths[i])
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        
        self.size = len(self.images)
    
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR)
        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR) / 255
 
        slic = Slic(num_components=config.TRAIN['num_superpixels'], compactness=10)
        SS_map = slic.iterate(np_img)
        
        SS_map = SS_map + 1

        SS_maps = []
        SS_maps_label = []

        for i in range(1, config.TRAIN['num_superpixels'] + 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    SS_maps_label.append(1)
                else:
                    SS_maps_label.append(0)
            else:
                SS_maps_label.append(0)

            SS_maps.append(buffer)
        
        ss_map = np.array(SS_maps)
        ss_maps_label = np.array(SS_maps_label)

        image = self.transform(image)
        depth = self.binary_loader(self.depths[index])

        depth = Image.fromarray(255 - np.array(depth))

        np_depth = np.array(depth)
        np_depth = cv2.resize(np_depth, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR)
        np_depth = cv2.cvtColor(np_depth, cv2.COLOR_GRAY2BGR)

        depth_SS_map = slic.iterate(np_depth)

        depth_SS_map = depth_SS_map + 1

        depth_SS_maps = []
        depth_SS_maps_label = []

        for i in range(1, config.TRAIN['num_superpixels'] + 1):
            buffer = np.copy(depth_SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    depth_SS_maps_label.append(1)
                else:
                    depth_SS_maps_label.append(0)
            else:
                depth_SS_maps_label.append(0)

            depth_SS_maps.append(buffer)
        
        depth_ss_map = np.array(depth_SS_maps)
        depth_ss_maps_label = np.array(depth_SS_maps_label)

        depth = self.depths_transform(depth)
        
        name = self.images[index].split('/')[-1]
        valid_name = self.images[index].split('/')[-3]
        
        image_for_post = self.rgb_loader(self.images[index])
        image_for_post = image_for_post.resize((self.testsize, self.testsize))
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        info = [gt.size, valid_name, name]
        
        gt = self.gt_transform(gt)
        
        return image, gt, depth, info, np.array(image_for_post), ss_map, ss_maps_label, depth_ss_map, depth_ss_maps_label

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')
    
    def __len__(self):
        return self.size


def get_testloader(val_image_root, valid_list, batchsize, testsize, shuffle=False, num_workers=12, pin_memory=False):

    dataset = SalObjDataset_test(val_image_root, valid_list, testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader