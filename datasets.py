import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import segmentation_transforms as seg_transforms
from matplotlib import pyplot as plt
import cv2

import CONSTS

class LoadMIRCInterp(data.Dataset):
    def __init__(self, imgpath, labelpath, num_parts, data_aug=False, orig_img_size=False):

        self.input_size = (108, 108)    # this size fits the gt interpretation. Cannot be (224, 224)
        self.orig_img_size = orig_img_size
        self.imgpath = imgpath
        self.labelpath = labelpath
        self.num_parts = num_parts
        self.image_filenames = self.make_dataset(imgpath)

        # txt = open(r'same_filenames.txt', 'w')
        # for filename in ds_same:
        #     txt.write(filename + '\n')
        #self.label_filenames = [os.path.join(labelpath, os.path.basename(f)) for f in self.image_filenames]

        if data_aug:
            self.transformer = self.get_transform()
        else:
            self.transformer = None

        # class ratio:
        self.inds_type0_examples = []
        self.inds_type1_examples = []
        #self.is_type1 = [False] * len(self.image_filenames)
        for ii, img_filename in enumerate(self.image_filenames):
            if os.path.exists(os.path.join(self.labelpath, os.path.basename(img_filename))):
                self.inds_type1_examples.append(ii)     # += 1
                #self.is_type1[ii] = True
            else:
                self.inds_type0_examples.append(ii)     # += 1
        self.class_ratio = [len(self.inds_type0_examples) / len(self.image_filenames),
                            len(self.inds_type1_examples) / len(self.image_filenames)]

    def __len__(self):
        return len(self.image_filenames)

    def get_transform(self):
        base_size = 108
        crop_size = 108

        min_size = 0.75 * base_size     #int((0.5 if train else 1.0) * base_size)
        max_size = 1.5 * base_size   #int((2.0 if train else 1.0) * base_size)
        transforms = []
        transforms.append(seg_transforms.RandomResize(min_size, max_size))
        #transforms.append(seg_transforms.RandomHorizontalFlip(0.5))
        transforms.append(seg_transforms.ColorJitter(hue=.05, saturation=.05))
        transforms.append(seg_transforms.RandomRotate(degrees=(-20, 20), fill_with=0))
        transforms.append(seg_transforms.RandomCrop(crop_size, fill_with=0))
        # transforms.append(seg_transforms.ToTensor())
        # transforms.append(seg_transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                               std=[0.229, 0.224, 0.225]))

        return seg_transforms.Compose(transforms)

    def __getitem__(self, index):

        # Image:
        img_filename = self.image_filenames[index]
        image = Image.open(img_filename)
        if not self.orig_img_size:
            image = image.convert('RGB').resize(self.input_size)

        is_transformed = False

        # Interp:
        lbl_filename = os.path.join(self.labelpath, os.path.basename(img_filename))
        if os.path.exists(lbl_filename):
            interp = cv2.imread(lbl_filename)[:, :, 0]
            #interp = Image.open(lbl_filename).convert('RGB')#.resize(self.input_size)
            ## Thicker contours
            thickness = 1
            for cls_ind in range(1, self.num_parts + 1, 1):     # use [5] if for EYE only
                ex, ey = np.nonzero(interp == cls_ind)
                for ii in range(-thickness, thickness, 1):
                    for jj in range(-thickness, thickness, 1):
                        interp[ex + ii, ey + jj] = cls_ind

            label = torch.tensor(1).long()

            # if adding data augmentation
            if self.transformer is not None:
                if np.random.rand() > 0.5:
                    image, interp = self.transformer(image=image, target=Image.fromarray(interp))
                    interp = np.array(interp)
                    is_transformed = True

        else:
            interp = np.random.randint(self.num_parts+1, size=self.input_size)  # random noise
            #random_interp_map = np.random.randint(self.num_parts+1, size=self.input_size)
            #interp = Image.fromarray(np.stack((random_interp_map, random_interp_map, random_interp_map), axis=2))
            label = torch.tensor(0).long()

        # Transform to torch tensors:
        norm_image = TF.to_tensor(image)

        if self.orig_img_size:
            norm_image = TF.normalize(norm_image, (0.5), (0.5))
        else:
            norm_image = TF.normalize(norm_image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        #interp = np.array(interp)[:, :, 0]
        interp = torch.tensor(interp).long()

        return (norm_image,
                interp,
                label,
                img_filename,
                is_transformed)

    def make_dataset(self, dirs):
        nparrays = []
        for dir in dirs:
            for fname in Path(dir).glob('**/*.png'):
                nparrays.append(str(fname))

        return nparrays

class datas(object):
    def __init__(self, trainset, testset, testvalset=None, testvalset2=None, num_parts=0):
        self.trainset = trainset
        self.testset = testset
        self.testvalset = testvalset
        self.testvalset2 = testvalset2
        self.num_parts = num_parts


# ===================
# load:
# ===================

def load_dataset(ds_name, negs_set=1, include_trainset=True, data_aug=False, keep_orig_input_size=False):

    data_dir_name =os.path.join(CONSTS.MI_DIR, 'imgs')
    labels_dir_name = os.path.join(CONSTS.MI_DIR, 'labels')
    negatives_dir_name = CONSTS.NEGATIVES_DIR
    negs_set = str(negs_set)
    if ds_name == 'HorseHead':
        category_folder_name = 'HORSE_HEAD'
        negative_folder_name = 'nonhorse'
        category_num_parts = 5
    elif ds_name == 'ManInSuit':
        category_folder_name = 'MAN_IN_SUIT'
        negative_folder_name = 'nonperson'
        category_num_parts = 7
    elif ds_name == 'HumanEye':
        category_folder_name = 'HUMAN_EYE'
        negative_folder_name = 'nonperson'
        category_num_parts = 6
    else:
        print('ERROR: dataset name does not exist..')
        return

    human_hardneg_folder = os.path.join(data_dir_name, 'hardneg_' + category_folder_name)
    dnn_hardneg_folder = os.path.join(negatives_dir_name, negative_folder_name + '_dnn_hardnegs')
    labels_folder = os.path.join(labels_dir_name, category_folder_name)

    train_images_path = [os.path.join(data_dir_name, category_folder_name, 'train'),
                         #os.path.join(human_hardneg_folder, 'train')
                         os.path.join(negatives_dir_name, negative_folder_name, negs_set)
                         ]

    test_images_path = [os.path.join(data_dir_name, category_folder_name, 'validation'),
                        #os.path.join(human_hardneg_folder, 'train')
                        os.path.join(negatives_dir_name, negative_folder_name, '0')]

    testval_images_path = [os.path.join(data_dir_name, category_folder_name, 'test'),
                           os.path.join(dnn_hardneg_folder, '4_2')]

    testval2_images_path = [os.path.join(data_dir_name, category_folder_name, 'test'),
                           os.path.join(human_hardneg_folder, 'test')]

    if not include_trainset:
        train_images_path = testval2_images_path

    ds = datas(LoadMIRCInterp(train_images_path, labels_folder, num_parts=category_num_parts, data_aug=data_aug, orig_img_size=keep_orig_input_size),
               LoadMIRCInterp(test_images_path, labels_folder, num_parts=category_num_parts, data_aug=False, orig_img_size=keep_orig_input_size),
               LoadMIRCInterp(testval_images_path, labels_folder, num_parts=category_num_parts, data_aug=False, orig_img_size=keep_orig_input_size),
               LoadMIRCInterp(testval2_images_path, labels_folder, num_parts=category_num_parts, data_aug=False, orig_img_size=keep_orig_input_size),
               num_parts=category_num_parts)

    print('loading dataset : %s.. number of train examples is %d, number of test examples is %d, number of testval examples is %d.'
          % (ds_name, len(ds.trainset), len(ds.testset), len(ds.testvalset)))

    return ds


if __name__ == '__main__':

    from utils.interpretation_utils import to_numpy, im_to_numpy, denormalize, plot_interpretation
    ds = load_dataset('HorseHead', negs_set=0, data_aug=True)
    fig = plt.figure(figsize=(16, 10))
    for k in range(10):  #len(ds.trainset)):
        img, interp, label, filename, is_transformed = ds.trainset.__getitem__(k)
        img = denormalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img, interp, label = im_to_numpy(img), to_numpy(interp), to_numpy(label)
        fig.clf()
        fig = plot_interpretation(fig, img=img, interp=interp, label=label, title=filename)
        fig.savefig('%d_t.png' % k)

    #
    # ds = load_dataset('ManInSuit')

    # for set_ in [ds.trainset, ds.testset, ds.testvalset]:
    #     img, interp, label, filename, is_transformed = set_.__getitem__(23)
    #     img, interp, label = im_to_numpy(img), to_numpy(interp), to_numpy(label)
    #     fig.clf()
    #     fig = plot_interpretation(fig, img=img, interp=interp, label=label, title=filename)
    #     fig.show()

        # img = denormalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # im = TorchToPIL(img)
        # im.show()
        #print(filename)
        # print(label)
        #time.sleep(2)

