from re import I
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb, cv2, six, os, sys, random
# cv2.setNumThreads(0)
import os.path as osp
import PIL
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
from glob import glob
import torchvision
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import warnings

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # import pdb; pdb.set_trace()
        img = img.resize(self.size, self.interpolation)
        # img.save('./2.png')
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class LMDBDataset_test(Dataset):
    def __init__(self, root=None, transform=None):
        
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=1099511627776)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples
        self.root = root
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # print('a')
        assert index <= len(self), 'index range error'
        index += 1
        # import pdb;pdb.set_trace()
        with self.env.begin(write=False) as txn:
            
            img_key = 'image-%09d' % index
            label_key = 'label-%09d' % index
            imgbuf = txn.get(img_key.encode())
            # end = time.time()
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label = str(txn.get(label_key.encode()).decode())
        
        if self.transform:
            img = self.transform(img)
        label = int(label)
        return (img, label)



class LMDBDataset(Dataset):
    def __init__(self, root=None, transform=None, aug=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples
            
        self.root = root
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            label_key = 'label-%09d' % index
            imgbuf = txn.get(img_key.encode())
            # end = time.time()
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
                # img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            # p_t = (time.time()-end)*100000
            # print(p_t)
            label = str(txn.get(label_key.encode()).decode())
        
        if self.transform:
            img = self.transform(img)

        label = int(label)
        return (img, label)

    def collate_fn(self, batch):
        # data augmentation
        self.seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            # iaa.CoarseDropout(0.01, size_percent=0.2),
            # iaa.MedianBlur(k=(3, 3)),
            ])
        img, label = zip(*batch)
        img = np.stack([i for i in img], axis=0)
        label = np.stack([i for i in label], axis=0)

        if self.aug:
            img = self.seq.augment_images(img)

        # Normalize
        img = img/255
        img = img-0.5
        # img = (img - 0.8412)/0.3176
        img = img[:,np.newaxis,:,:]
        return (img, label)
    
if __name__ == '__main__':
    char2idx_dict = OrderedDict()
    idx2char_dict = OrderedDict()
    

    char2index = {}
    f = open('./char_book2024.txt', 'r').readlines()
    for i, char in enumerate(f):
        char = char.strip()
        char2idx_dict[i] = char


    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.ToTensor(),
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                #    transforms.ToTensor(),
                                #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}
    
    from composer.algorithms.randaugment import RandAugmentTransform 
    randaugment_transform = RandAugmentTransform(severity=9,
                                                depth=2,
                                                augmentation_set="all")
    composed = transforms.Compose([transforms.ToTensor(),
                                #    randaugment_transform
                                   ])
    dataset = LMDBDataset(
        # root='/home/zyy/SLC/SLC_char_reg/dataset/train/infer_img_lmdb/char_infer_cmi_normal_thin2_9w_lmdb',
        root='/home/zyy/SLC/SLC_char_reg/dataset/test/SLC_handw_collect_allinformation/2_1206_all_lmdb',
        aug=True,
        transform=composed
    )
    
    
    # dataset = LMDBDataset_test(
    #     root='/home/zyy/SLC/SLC_char_reg/dataset/test/SLC_handw_collect_allinformation/2_1206_all_lmdb',
    #     # root='/home/zyy/SLC/SLC_char_reg/dataset/test/test_true_char_im_guji_10_lmdb',
    #     transform=transforms.Compose([
    #         # transforms.Resize((96,96)),
    #         transforms.ToTensor(),
    #                                ])
    # )

    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        sampler=None, 
        drop_last=True, 
        pin_memory=True,
        num_workers=0, 
        shuffle=False,
        # collate_fn=my_collate_fn
    )
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    for i, data in enumerate(train_loader):

        label = data[1]
        label = char2idx_dict[int(label)]
        img = data[0]
        torchvision.utils.save_image(img, f'/home/zyy/SLC/SLC_char_reg/debug/{label}.jpg', normalize=True)
        # import pdb;pdb.set_trace()
        pass
