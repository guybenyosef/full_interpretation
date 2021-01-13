import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.interpretation_utils import to_numpy, compute_auc_multiclass, compute_iou_interpretation

########################################
########################################
# Data
########################################
########################################
def sample_dataset(trainset, valset, testset, overfit, subset, batch_size, num_workers, verbose=False):
    if overfit:  # sample identical very few examples for both train ans val sets:
        num_samples_for_overfit = 10
        type1 = np.random.choice(trainset.inds_type1_examples, num_samples_for_overfit)
        type0 = np.random.choice(trainset.inds_type0_examples, num_samples_for_overfit)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                  shuffle=False, pin_memory=True)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                 shuffle=False, pin_memory=True)
        nott = ("DATA: Sampling identical sets of %d POS and %d NEG examples for train and val sets.. " % (num_samples_for_overfit, num_samples_for_overfit))

    elif subset is not None:
        # Train:
        type1 = np.asarray(trainset.inds_type1_examples)  # all pos
        type0 = np.random.choice(trainset.inds_type0_examples, subset)  # subset neg
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                  shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)
        # Test:
        type1 = np.asarray(valset.inds_type1_examples)  # all pos
        if subset > 0:
            type0 = np.asarray(valset.inds_type0_examples)  # all neg
        else:
            type0 = np.array([]).astype(int)  # 0 neg
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                 shuffle=False, pin_memory=True,
                                                 num_workers=num_workers)
        nott = ("DATA: Sampling all POS and %d NEG examples for train and val sets.. " % subset)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True,
                                                  num_workers=num_workers)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 shuffle=True, pin_memory=True,
                                                 num_workers=num_workers)
        nott = ("DATA: Sampling all POS and all NEG examples for train and val sets.. ")

    # load test
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, pin_memory=True,
                                            num_workers=num_workers)

    if verbose:
        print(nott)

    return trainloader, valloader, testloader

########################################
########################################
# Run epoch for Train/Validate/Eval
########################################
########################################
def run_epoch(epoch, dataset, optimizer, model, criterion, device, prefix='Training', returned_examples=0):

    # Setting prefix:
    if optimizer is not None:
        model.train()
        #prefix = 'Training'
    else:
        model.eval()
        #prefix = 'Validating'

    # Init:
    total_loss, total_accuracy, total_loss_class, total_loss_interp = 0.0, 0.0, 0.0, 0.0
    total_predicted_class, total_predicted_interp, total_gt_class, total_gt_interp, total_filenames = [], [], [], [], []
    iou_batches = []

    with tqdm(total=len(dataset), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:

        for batch_idx, batch_data in enumerate(dataset):

            # zero the parameter gradients
            if optimizer is not None:
                model.zero_grad()

            # Extract batch data:
            if device.type == 'cpu':
                images, interp_gt, class_gt = batch_data[:3]
            else:
                images, interp_gt, class_gt = [d.cuda() for d in batch_data[:3]]
            filenames = batch_data[3]
            is_transformed = batch_data[4]

            # =================== forward =====================
            if optimizer is not None:
                images.requires_grad_()

            if model.output_type == 'img2class':
                class_out = model(images)
                interp_out = interp_gt  # not used:
                loss_class = criterion(class_out, class_gt)
                loss = loss_class
                total_loss_class += loss_class.item() * len(class_gt)
                total_loss_interp += 0

            elif model.output_type == 'img2intep':
                relevant_examples_in_batch = torch.where(class_gt > 0)
                if len(relevant_examples_in_batch[0]) > 0:
                    # remove non-class examples, which are not relevant to interpretation-only task:
                    images = images[relevant_examples_in_batch]
                    interp_gt = interp_gt[relevant_examples_in_batch]
                    class_gt = class_gt[relevant_examples_in_batch]
                    filenames = tuple([filenames[i] for i in relevant_examples_in_batch[0].tolist()])
                    # Continue with standard pipeline:
                    class_out = torch.tensor([[0, 1]] * len(class_gt), device=device)  # class_gt # not used:
                    interp_out = model(images)
                    interp_array_loss = criterion(interp_out, interp_gt) #convert_onehot(interp_gt, model.num_parts)
                    if device.type != 'cpu':
                        interp_array_loss = interp_array_loss.cuda()
                    sum_loss_interp = torch.sum(interp_array_loss * class_gt)
                    loss_interp = sum_loss_interp / torch.sum(class_gt)
                    loss = loss_interp
                    total_loss_interp += sum_loss_interp.item()     # * len(class_gt)
                    total_loss_class += 0
                else:
                    loss = None

            elif model.output_type == 'img2dual':
                class_out, interp_out = model(images)
                # -- process interp:
                interp_array_loss = criterion[0](interp_out, interp_gt)
                if device.type != 'cpu':
                    interp_array_loss = interp_array_loss.cuda()
                sum_loss_interp = torch.sum(interp_array_loss * class_gt)
                sum_class_gt = torch.sum(class_gt)
                if sum_class_gt > 0:
                    loss_interp = sum_loss_interp / sum_class_gt
                else:
                    loss_interp = sum_loss_interp
                # -- process class:
                class_array_loss = criterion[1](class_out, class_gt)
                if device.type != 'cpu':
                    class_array_loss = class_array_loss.cuda()
                    is_transformed = is_transformed.cuda()
                sum_loss_class = torch.sum(class_array_loss * ~is_transformed)
                sum_is_transformed = torch.sum(~is_transformed)
                if sum_is_transformed > 0:
                    loss_class = sum_loss_class / sum_is_transformed
                else:
                    loss_class = sum_loss_class
                # -- combine:
                loss = model.loss_ratio[0] * loss_interp + model.loss_ratio[1] * loss_class
                total_loss_interp += sum_loss_interp.item()
                total_loss_class += sum_loss_class.item()


            if loss is not None:
                # do backward:
                total_loss += loss.item() * len(class_gt)
                if optimizer is not None: # in train mode
                    loss.backward()
                    optimizer.step()

                # Compute accuracy and/or other metrics:
                _, predicted_class = torch.max(class_out, 1)
                total_accuracy += (predicted_class == class_gt).float().sum().item()
                predicted_class = to_numpy(predicted_class)
                total_predicted_class = total_predicted_class + list(predicted_class)
                class_gt = to_numpy(class_gt)
                total_gt_class = total_gt_class + list(class_gt)

                # TODO Dice (uncomment below)
                # interp_out = torch.sigmoid(interp_out)

                # Detach
                class_out = to_numpy(class_out)
                interp_out = to_numpy(interp_out)
                interp_gt = to_numpy(interp_gt)
                iou_batch = compute_iou_interpretation(predicted_interp=interp_out, groundtruth_interp=interp_gt, num_parts=model.num_parts)
                # TODO Dice (comment below)
                iou_batch = iou_batch[:, 1:] # remove the background class

                iou_batches.append(iou_batch[np.where(class_gt)])

                # calc amount of returned interp results examples:
                if returned_examples > 0:
                    still_to_add = returned_examples - len(total_gt_interp)
                    if still_to_add > 0:
                        still_to_add_in_batch = min(still_to_add, returned_examples)
                        if still_to_add_in_batch > 0:
                            total_gt_interp = total_gt_interp + list(interp_gt[:still_to_add_in_batch])
                            total_predicted_interp = total_predicted_interp + list(interp_out[:still_to_add_in_batch])

                total_filenames = total_filenames + list(filenames)
                images = to_numpy(images)
                del loss
                if 'loss_class' in locals():
                    del loss_class
                if 'loss_interp' in locals():
                    del loss_interp

            # Update pbar:
            pbar.update()

    # Returned metrics:
    auc = compute_auc_multiclass(predicted_labels=total_predicted_class, groundtruth_labels=total_gt_class, max_num_labels=2)
    iou = np.array([0] * model.num_parts)
    if len(iou_batches) > 0:
        iou = np.mean(np.concatenate(iou_batches), axis=0)
    acc = 0
    if total_accuracy > 0:
        acc = total_accuracy / len(total_filenames)

    mean_total_loss, mean_total_loss_interp, mean_total_loss_class = 0.0, 0.0, 0.0
    if len(total_filenames) > 0:
        mean_total_loss, mean_total_loss_interp, mean_total_loss_class = \
            total_loss / len(total_filenames), total_loss_interp / len(total_filenames), total_loss_class / len(total_filenames)

    misc = [total_predicted_class, total_gt_class, total_filenames, total_predicted_interp, total_gt_interp]

    return mean_total_loss, mean_total_loss_interp, mean_total_loss_class, acc, auc, iou, misc

