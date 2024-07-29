# coding=utf-8
import os.path
import numpy as np
from mxnet.gluon.data.vision import datasets
from mxnet import image, nd


def img_normalization(img):
    img = img.astype('float32') / 255
    normalized_img = image.color_normalize(img, mean=nd.array([0.485, 0.456, 0.406]),
                                           std=nd.array([0.229, 0.224, 0.225]))
    return normalized_img


class MultiViewImageDataset(datasets.ImageFolderDataset):
    def __init__(self, root, num_view, flag=1, transform=None):
        super(MultiViewImageDataset, self).__init__(root, flag, transform)
        self._num_view = num_view
        self.num_classes = len(self.synsets)

    def __len__(self):
        return len(self.items) // self._num_view

    def __getitem__(self, idx):
        imgs = []
        num = ''
        depths_7 = []
        depths_14 = []
        depths_28 = []
        loader_7 = 1
        loader_14 = 1
        loader_28 = 0
        for item in self.items[idx * self._num_view:idx * self._num_view + self._num_view]:
            listpath = str(item[0]).split('/')
            a = os.path.split(str(item[0]))
            b = os.path.abspath(os.path.join(a[0], '../../..'))
            name = listpath[-1].split('.')[0]
            num = listpath[-2]
            img = image.imread(item[0], self._flag)
            if self._transform is not None:
                img = self._transform(img)
            imgs.append(img.expand_dims(0))
            if loader_7 == 1:
                depth_path_7 = os.path.join(b, 'index_7_36', listpath[-3], listpath[-2], name)
                depth_path_7 = depth_path_7 + '.npy'
                depth_7 = np.load(depth_path_7)
                depth_7 = nd.array(depth_7)
                depths_7.append(depth_7.expand_dims(0))
            if loader_14 == 1:
                depth_path_14 = os.path.join(b, 'index_14_36', listpath[-3], listpath[-2], name)
                depth_path_14 = depth_path_14 + '.npy'
                depth_14 = np.load(depth_path_14)
                depth_14 = nd.array(depth_14)
                depths_14.append(depth_14.expand_dims(0))
            if loader_28 == 1:
                depth_path_28 = os.path.join(b, 'index_28_36', listpath[-3], listpath[-2], name)
                depth_path_28 = depth_path_28 + '.npy'
                depth_28 = np.load(depth_path_28)
                depth_28 = nd.array(depth_28)
                depths_28.append(depth_28.expand_dims(0))
        label = self.items[idx * self._num_view][1]
        ##0
        if loader_7 == 0 and loader_14 == 0 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), label
        ##1
        if loader_7 == 1 and loader_14 == 0 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), label

        if loader_7 == 0 and loader_14 == 1 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_14, dim=0), label

        if loader_7 == 0 and loader_14 == 0 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_28, dim=0), label
        ##2
        if loader_7 == 1 and loader_14 == 1 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), nd.concat(*depths_14, dim=0), label

        if loader_7 == 1 and loader_14 == 0 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), nd.concat(*depths_28, dim=0), label

        if loader_7 == 0 and loader_14 == 1 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_14, dim=0), nd.concat(*depths_28, dim=0), label
        ##3
        if loader_7 == 1 and loader_14 == 1 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), nd.concat(*depths_14, dim=0), nd.concat(
                *depths_28, dim=0), label
