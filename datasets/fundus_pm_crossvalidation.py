import cv2
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import numpy as np
from skimage.transform import resize
from PIL import Image

class traindataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None, train=True, args=None):
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
        self.multiaug = args.multiaug
        self.multitask = args.multitask


        images_path = np.genfromtxt( self.root_dir + '/PAML/random_list.txt', dtype='str')
        images_path = list(images_path)
        images_path = [self.root_dir + item for item in images_path]

        num_fold = int(len(images_path) / 5)
        if args.seed == 0:
            test_path = images_path[:num_fold]
        elif args.seed == 1:
            test_path = images_path[num_fold:2*num_fold]
        elif args.seed == 2:
            test_path = images_path[2 * num_fold:3 * num_fold]
        elif args.seed == 3:
            test_path = images_path[3 * num_fold:4 * num_fold]
        elif args.seed == 4:
            test_path = images_path[4 * num_fold:5 * num_fold]

        train_path = list(set(images_path) - set(test_path))

        label_list_train = [1 if item.split("/")[-1][0] == "P" else 0 for item in train_path]
        label_list_test = [1 if item.split("/")[-1][0] == "P" else 0 for item in test_path]

        if self.train:
            self.train_dataset = []
            self.targets = []
            self.rotation_label = []
            for i in range(0, len(train_path)):
                image = cv2.imread(args.data + "/PAML/resized_image_320/" + train_path[i].split("/")[-1])
                self.train_dataset.append(image)
                self.targets.append(label_list_train[i])
                self.name.append(train_path[i].split("/")[-1])
                self.rotation_label.append(0)
            print("Train images PM ", len(self.train_dataset), "P: ", sum(self.targets), "Neg: ", len(self.targets) - sum(self.targets))
        else:
            self.train_dataset = []
            self.targets = []
            for i in range(0, len(test_path)):
                image = cv2.imread(args.data + "/PAML/resized_image_320/" + test_path[i].split("/")[-1])
                self.train_dataset.append(image)
                self.targets.append(label_list_test[i])
                self.name.append(test_path[i].split("/")[-1])
            print("Test images PM ", len(self.train_dataset), "P: ", sum(self.targets), "Neg: ", len(self.targets) - sum(self.targets))

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):

        sample = self.train_dataset[idx]

        sample = Image.fromarray(np.uint8(sample))

        img = self.transform(sample)
        target = self.targets[idx]

        if self.train and self.multiaug:
            img2 = self.transform(sample)
            if self.multitask:
                rotation_label = self.rotation_label[idx]
                return [img, img2], [target, rotation_label], idx, self.name[idx]
            else:
                return [img, img2], target, idx, self.name[idx]

        return img, target, idx, self.name[idx]

if __name__ == '__main__':
    count = 0
    tot_count = 0
