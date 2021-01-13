import numpy as np
import torch
import torchvision
import cv2
from sklearn import metrics
from torch.utils.tensorboard.summary import hparams
from PIL import Image, ImageDraw
import torch.nn.functional as F


#########################
#########################
# utils Handling IMAGES
#########################
#########################

# define for later use:
TorchToPIL = torchvision.transforms.ToPILImage()
PILtoTorch = torchvision.transforms.ToTensor()
Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# from: https://github.com/namedBen/Convolutional-Pose-Machines-Pytorch/blob/master/train_val/Mytransforms.py
def normalize(tensor, mean, std):
    """Normalize a ``torch.tensor``
    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: Normalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def denormalize(tensor, mean, std):
    """Normalize a ``torch.tensor``
    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: deNormalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def convert_onehot(seg_mask, num_classes):
    return torch.transpose(F.one_hot(seg_mask, num_classes=num_classes), 1, 3)


#########################
#########################
# Plot utils
#########################
#########################

def plot_output_img2type(fig, inputs, labels, outputs, panels_inds, _prefix, set_indices=range(4)):
    # extract original dot image:
    input_images = inputs.cpu().data

    # extract output read:
    out_type = outputs.cpu().data.numpy()

    # extract gt :
    gt_type = labels.cpu().data.numpy()

    # plot:
    num_examples_to_plot = 4

    for indx in range(min(input_images.size(0), num_examples_to_plot)):
        ax = fig.add_subplot(2, 4, panels_inds[indx])
        img = TorchToPIL(input_images[set_indices[indx]])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title('(%s) out: %.2f, gt: %.2f' % (_prefix, np.argmax(out_type[indx]), np.argmax(gt_type[indx])))

    return fig


def plot_test_img2type(fig, filenames, gt_same, pred_same):
    # plot:
    num_examples_to_plot = 16

    for indx in range(min(len(filenames), num_examples_to_plot)):
        ax = fig.add_subplot(4, 4, indx + 1)
        img = cv2.imread(filenames[indx])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title('pred: %.2f, gt: %.2f' % (pred_same[indx], gt_same[indx]))

    return fig


def plot_maps(fig, img, interp, label, class_out, maps, single_img_flag=False):
    '''plot feature maps and detected segmentation map
    Input:
        fig         : figure object <Matplotlib.pyplot figure>
        img         : input image <numpy.ndarray, (num_classes,row,col, 3)>
        maps         : maps from model <numpy.ndarray, (num_classes,row,col)>

    Output:
        segmap       : final segmentation <numpy.ndarray, (img.h,img.w)>
    '''

    num_classes = len(maps)
    #img = np.transpose(img, (1, 2, 0))  # H*W*C

    seg = np.argmax(maps, axis=0)

    # TODO DICE
    # prob_class, predicted_class = np.max(maps, 1)  # for dice
    # predicted_class = torch.where(prob_class < 0.5, predicted_class, np.zeros(predicted_class.size(), dtype=int))

    if single_img_flag:
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img)
        imm = ax.imshow(seg, cmap='jet', alpha=0.4)
        ax.set_axis_off()
        fig.colorbar(imm)

    else:
        ax = fig.add_subplot(1, num_classes+2, 1)
        ax.imshow(img)
        ax.imshow(seg, cmap='jet', alpha=0.4)
        ax.set_title('pred: %.1f' % np.argmax(class_out))
        ax.set_axis_off()

        ax = fig.add_subplot(1, num_classes+2, 2)
        ax.imshow(img)
        imm = ax.imshow(interp, cmap='jet', alpha=0.4)
        ax.set_axis_off()
        ax.set_title('gt: %.1f' % label)
        fig.colorbar(imm)
        #
        for map_indx in range(num_classes):
            ax = fig.add_subplot(1, num_classes+2, map_indx+3)
            ax.imshow(maps[map_indx], extent=[0, 1, 0, 1])
            ax.set_title("c=%d" % map_indx)
            ax.set_axis_off()

    return fig

def plot_interpretation(fig, img, interp, label, title=''):
    '''plot feature maps and detected segmentation map
    Input:
        fig         : figure object <Matplotlib.pyplot figure>
        img         : input image <numpy.ndarray, (num_classes,row,col, 3)>
        maps         : maps from model <numpy.ndarray, (num_classes,row,col)>

    Output:
        segmap       : final segmentation <numpy.ndarray, (img.h,img.w)>
    '''

    #img = np.transpose(img, (1, 2, 0))  # H*W*C

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.set_title(title)
    ax.set_axis_off()

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    imm = ax.imshow(interp, cmap='jet', alpha=0.4)
    ax.set_title('gt: %.1f' % label)
    ax.set_axis_off()
    fig.colorbar(imm)

    return fig


def features2seg(maps):
    return np.argmax(maps)


def plot_misclassifications_grid(files, labels):

    new_im = Image.new('RGB', (410, 410))

    index = 0
    idddxxx = np.random.randint(0, len(files), min(100, len(files)))
    for i in range(10, 400, 40):
        for j in range(10, 400, 40):
            if index < len(files):
                im = Image.open(files[idddxxx[index]])
                lb = labels[idddxxx[index]]
                im.thumbnail((30, 30))
                draw = ImageDraw.Draw(im)
                #draw.text((0, 0), str(lb), fill=128)
                draw.text((0, 0), str(lb), fill=0) #color=(255, 0, 255))#'magenta')
                new_im.paste(im, (i, j))
                index += 1


    #new_im.show()
    #input("Press Enter to continue...")

    return new_im


#########################
#########################
# Log utils
#########################
#########################

def better_hparams(writer, hparam_dict=None, metric_dict=None):
    """Add a set of hyperparameters to be compared in TensorBoard.
    Args:
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
          name of the hyper parameter and it's corresponding value.
        metric_dict (dictionary): Each key-value pair in the dictionary is the
          name of the metric and it's corresponding value. Note that the key used
          here should be unique in the tensorboard record. Otherwise the value
          you added by `add_scalar` will be displayed in hparam plugin. In most
          cases, this is unwanted.

        p.s. The value in the dictionary can be `int`, `float`, `bool`, `str`, or
        0-dim tensor
    Examples::
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter() as w:
            for i in range(5):
                w.add_hparams({'lr': 0.1*i, 'bsize': i},
                              {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
    Expected result:
    .. image:: _static/img/tensorboard/add_hparam.png
       :scale: 50 %
    """
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    # writer.file_writer.add_summary(sei)
    # for k, v in metric_dict.items():
    #     writer.add_scalar(k, v)
    # with SummaryWriter(log_dir=os.path.join(self.file_writer.get_logdir(), str(time.time()))) as w_hp:
    #     w_hp.file_writer.add_summary(exp)
    #     w_hp.file_writer.add_summary(ssi)
    #     w_hp.file_writer.add_summary(sei)
    #     for k, v in metric_dict.items():
    #         w_hp.add_scalar(k, v)

    return sei




#########################
#########################
# STATS
#########################
#########################


#############################
# Compute AUC (or AP)
#############################
def compute_auc_multiclass(predicted_labels, groundtruth_labels, max_num_labels=5):
    '''
    compute AUC
    :param W: <int> predicted_labels, <int> groundtruth_labels
    :return: <numpy> mean Average Precision (a.k.a., Area Under Curve)
    '''

    auc = np.full(max_num_labels, np.nan)

    for class_indx in range(1,5):
        if (0 in groundtruth_labels) and (class_indx in groundtruth_labels):
            noclass_inds = np.where(np.array(predicted_labels) == 0)[0]
            yesclass_inds = np.where(np.array(predicted_labels) == class_indx)[0]
            relevant_indices = np.concatenate((noclass_inds, yesclass_inds), axis=0)
            relevant_gt_labels = [0] * len(relevant_indices)
            relevant_outputs_class = [0] * len(relevant_indices)
            for ii in range(len(relevant_indices)):
                if groundtruth_labels[relevant_indices[ii]] > 0:
                    relevant_gt_labels[ii] = 1
                if predicted_labels[relevant_indices[ii]] > 0:
                    relevant_outputs_class[ii] = 1

            auc[class_indx] = metrics.roc_auc_score(y_true=relevant_gt_labels, y_score=relevant_outputs_class)
            #print("AUC [0 vs. %d] is %.6f" % (class_indx, auc[class_indx]))

    mAP = auc[~np.isnan(auc)].mean()

    #print("Mean Average Percision (mAP) is %.6f " % mAP)

    return mAP

#############################
# Compute IoU
#############################
def compute_iou_interpretation(predicted_interp, groundtruth_interp, num_parts=5+1):
    '''
    compute Intersection Over Union for interpretation
    :param: <numpy> predicted_interp, <numpy> groundtruth_interp, both in dimensions img_width*img_height and values between 0 to num_parts
    :return: <numpy> Intersection Over Union per part
    '''

    if predicted_interp.ndim == 4:
        predicted_interp = np.argmax(predicted_interp, axis=1)
        # TODO DICE (comment above and uncomment below)
        # prob_interp = np.max(predicted_interp, 1)  # for dice
        # predicted_interp = np.argmax(predicted_interp, 1)  # for dice
        # predicted_interp = np.where(prob_interp < 0.5, predicted_interp, np.zeros(predicted_interp.shape).astype(np.int))

    iou = np.zeros(num_parts)
    if predicted_interp.ndim > 2:
        num_batches = len(predicted_interp)
        iou = np.zeros(shape=(num_batches, num_parts))

    for cls_indx in range(num_parts):
        groundtruth_mask = (predicted_interp==cls_indx).astype(bool)
        predicted_mask = (groundtruth_interp==cls_indx).astype(bool)
        overlap = groundtruth_mask * predicted_mask  # Logical AND
        union = groundtruth_mask + predicted_mask  # Logical OR
        # intersect = np.dot(groundtruth_mask.reshape(-1), predicted_mask.reshape(-1))
        # union = np.sum(groundtruth_mask) + np.sum(predicted_mask)
        if predicted_interp.ndim == 2:
            if union.sum() > 0:
                iou[cls_indx] = overlap.sum() / float(union.sum())  # Treats "True
        if predicted_interp.ndim == 3:
            iou_with_possible_nan = overlap.sum(axis=(1, 2)) / union.sum(axis=(1, 2)) # NaN occurs where union is 0
            iou_with_possible_nan[np.isnan(iou_with_possible_nan)] = 0 # replace NaNs with 0
            iou[:, cls_indx] = iou_with_possible_nan


    return iou

#############################
# Main
#############################
if __name__ == '__main__':
    interp_pred = np.random.randint(low=0, high=6, size=(108, 108)).astype(np.long)
    interp_gt = np.random.randint(low=0, high=6, size=(108, 108)).astype(np.long)

    iou1 = compute_iou_interpretation(interp_pred, interp_pred)
    iou2 = compute_iou_interpretation(interp_pred, interp_gt)

    interp_pred = np.random.randint(low=0, high=6, size=(10, 108, 108)).astype(np.long)
    interp_gt = np.random.randint(low=0, high=6, size=(10, 108, 108)).astype(np.long)

    iou3 = compute_iou_interpretation(interp_pred, interp_pred)
    iou4 = compute_iou_interpretation(interp_pred, interp_gt)

    interp_pred = np.random.random(size=(10, 6, 108, 108))
    interp_gt = np.random.randint(low=0, high=6, size=(10, 108, 108)).astype(np.long)

    iou5 = compute_iou_interpretation(interp_pred, np.argmax(interp_pred, axis=1))
    iou6 = compute_iou_interpretation(interp_pred, interp_gt)


    print('done')
    pass


