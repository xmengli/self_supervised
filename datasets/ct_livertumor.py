import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from medpy.io import load, save
import numpy as np
from skimage.transform import resize
from PIL import Image
import time
import _pickle as cPickle
import torch
class traindataset(data.Dataset):
    def __init__(self, root, transform=None, train=True, test_type="val", args=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train_data = []
        self.targets = []
        self.train_name = []
        self.multiaug = args.multiaug
        self.multitask = args.multitask
        self.train = train
        self.test_type = test_type

        # train_id = [9, 8, 97, 30, 42, 39, 109, 71, 57, 110, 6, 55, 52, 53, 62, 114, 129, 32, 85, 113, 28, 86, 34, 51, 122, 127, 13, 49, 67, 25, 94, 23, 65, 5, 80, 73, 84, 111, 121, 81, 64, 21, 15, 24, 87, 4, 59, 104, 11, 66, 50, 101, 60, 29, 88, 103, 1, 91, 96, 26, 124, 130, 75, 18, 14, 41]
        # val_id   = [102, 22, 38, 72, 63, 123, 37, 0, 16, 3, 99, 83, 92, 43, 115, 46, 12, 19, 48, 89, 76, 120, 54, 82, 56, 116, 100, 95, 105, 61, 79, 77, 70]
        # test_id  = [106, 44, 45, 117, 27, 128, 90, 126, 20, 108, 7, 17, 78, 31, 118, 36, 47, 2, 74, 68, 10, 112, 35, 107, 93, 98, 69, 58, 125, 33, 40, 119]

        train_id = os.listdir(self.root + "lits/data_slices/train/")
        val_id  = os.listdir(self.root + "lits/data_slices/val/")
        test_id = os.listdir(self.root + "lits/data_slices/test/")

        time_init = time.time()
        if self.train:
            self.rotation_label = []
            for i in range(0, len(train_id)):
                image_name = self.root + "lits/data_slices/train/slices-"+str(i)+".pkl"
                self.train_data.append(image_name)
                self.train_name.append("train/slices-"+str(i)+".pkl")
                self.rotation_label.append(0)
            self.targets = np.loadtxt(self.root + '/lits/data_slices/train_label3class.txt', dtype="int")
            print ("Training Data: ", len(self.train_data), " Liver Scans")
            print ("time", time.time() - time_init)
        else:
            if self.test_type == "val":
                for i in range(0, len(val_id)):
                    image_name = self.root + "lits/data_slices/val/slices-"+str(i)+".pkl"
                    self.train_data.append(image_name)
                    self.train_name.append("val/slices-" + str(i) + ".pkl")
                self.targets = np.loadtxt(self.root + '/lits/data_slices/val_label3class.txt', dtype="int")
                print ("Val Data: ", len(self.train_data), " Liver Scans")
                print ("time", time.time() - time_init)
            else:
                for i in range(0, len(test_id)):
                    image_name = self.root + "lits/data_slices/test/slices-"+str(i)+".pkl"
                    self.train_data.append(image_name)
                    self.train_name.append("test/slices-" + str(i) + ".pkl")
                self.targets = np.loadtxt(self.root + '/lits/data_slices/test_label3class.txt', dtype="int")
                print ("Test Data: ", len(self.train_data), " Liver Scans")
                print ("time", time.time() - time_init)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        image_path, target  = self.train_data[index], self.targets[index]
        img = cPickle.load(open(image_path, "rb"))


        # normalize
        img = ((img - np.min(img))*255) / (float(np.max(img) - np.min(img)))

        img = Image.fromarray(img.astype("uint8"))
        img1 = self.transform(img)
        if self.train and self.multiaug:
            img2 = self.transform(img)
            if self.multitask:
                rotation_label = self.rotation_label[index]
                return [img1, img2], [target, rotation_label], index, self.train_name[index]
            else:
                return [img1, img2], target, index, self.train_name[index]

        return img1, target, index, self.train_name[index]



if __name__ == '__main__':
    count = 0
