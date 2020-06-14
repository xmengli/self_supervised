import cv2
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob

class traindataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None, train=True, test_type="amd", args=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root
        self.transform = transform
        self.name = []
        self.train = train
        self.multitask = args.multitask
        self.multiaug = args.multiaug
        self.test_type = test_type

        self.synthesis = args.synthesis
        self.train_syn  = []

        if self.train:
            train_path = list(np.genfromtxt(self.root_dir + "/kaggle_dr/data_list.txt", dtype='str'))
            self.train_dataset = [self.root_dir + "/kaggle_dr/"+ item for item in train_path]
            self.targets        = [0] * len(train_path) # did not load labels for training data.
            self.rotation_label = [0] * len(train_path)
            self.name = train_path
            print("train data: ", len(self.train_dataset))
            self.train_syn = ["../pytorch-CycleGAN-and-pix2pix-master/results/fundusFFA_cyclegan_lr/test_latest/" + item for item in train_path]
            self.train_syn = [item.replace(".jpeg",".png") for item in self.train_syn]
            print ("syn data", len(self.train_syn))
        else:
            if self.test_type == "amd":
                test_path_amd = list(np.genfromtxt(self.root_dir + '/Training400/random_list.txt', dtype='str'))
                test_path_amd = [item for item in test_path_amd if item.split("/")[-1] != "A0012.jpg"]
                self.train_dataset = [self.root_dir + "/Training400/resized_image_320/"+ item.split("/")[-1] for item in test_path_amd]
                self.targets =  [1 if item.split("/")[-1][0] == "A" else 0 for item in test_path_amd]
                self.name = test_path_amd
                print("Test images AMD ", len(self.train_dataset), "P: ", sum(self.targets), "N: ", len(self.targets) - sum(self.targets))
            elif self.test_type == "gon":
                test_path_gon = list(np.genfromtxt(self.root_dir + '/iChanllenge-Gon/Training400/random_index.txt', dtype='str'))
                self.train_dataset = [self.root_dir + "/iChanllenge-Gon/Training400/resized_images_320/" + item.split("/")[-1] for item in test_path_gon]
                self.targets = [1 if item.split("/")[-1][0] == "g" else 0 for item in test_path_gon]
                self.name = test_path_gon
                print("Test images GON ", len(self.train_dataset), "P: ", sum(self.targets), "N: ",
                      len(self.targets) - sum(self.targets))
            elif self.test_type == "pm":
                test_path_pm = list(np.genfromtxt( self.root_dir + '/PAML/random_list.txt', dtype='str'))
                self.train_dataset = [self.root_dir + "/PAML/resized_image_320/"+ item.split("/")[-1] for item in test_path_pm]
                self.name = test_path_pm
                self.targets = [1 if item.split("/")[-1][0] == "P" else 0 for item in test_path_pm]
                print("Test images PM ", len(self.train_dataset), "P: ", sum(self.targets), "N: ", len(self.targets) - sum(self.targets))


    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):

        sample = self.train_dataset[idx]
        sample = cv2.imread(sample)

        sample = Image.fromarray(np.uint8(sample))

        img = self.transform(sample)
        target = self.targets[idx]

        if self.train and self.multiaug:

            img2 = self.transform(sample)

            sample_syn = self.train_syn[idx]
            sample_syn = cv2.imread(sample_syn)
            sample_syn = Image.fromarray(np.uint8(sample_syn))
            img_syn = self.transform(sample_syn)
            img_syn2 = self.transform(sample_syn)

            img3 = self.transform(sample)
            return [img, img2, img3, img_syn], [target], idx, self.name[idx]

            # if self.multitask:
            #     rotation_label = self.rotation_label[idx]
            #     return [img, img2], [target, rotation_label], idx, self.name[idx]
            # else:
            #     return [img, img2], target, idx, self.name[idx]

        return img, target, idx, self.name[idx]

if __name__ == '__main__':
    count = 0
    tot_count = 0
