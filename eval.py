import torch
import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2

import CONSTS
from all_models import load_model
from all_losses import load_loss
from datasets import load_dataset
from utils.interpretation_utils import plot_maps, plot_misclassifications_grid
from sacred import Experiment
from torch.utils.data import DataLoader
from engine import run_epoch

ex = Experiment('SemanticSupport')

@ex.config
def config():
    batch_size = None  # 'Specify the number of batches'
    #weights = 'storage_unix/logs/HorseHead/RecUNetMirc/[1.0, 10.0]/5/weights_HorseHead_RecUNetMirc_best.pth'
    weights = 'storage/logs/HorseHead/RecUNetMirc/[1.0, 1.0]/17/weights_HorseHead_RecUNetMirc_best.pth'#storage_unix2/logs/HorseHead/ResNet/[1.0, 1.0]/4/weights_HorseHead_ResNet_best.pth'
    #weights = 'storage/logs/HorseHead/ResNet/[1.0,1.0]/1/weights_HorseHead_ResNet_best.pth'# 'The absolute path of the weights'
    gpu = 0  # 'Select visible gpu device [0-3]. If not gpu then use -1.'
    subset = None
    dataset = 'HorseHead'
    num_workers = 32
    plot_flag = False
    num_examples_to_plot = 0
    testid = '0'

@ex.named_config
def mac():
    batch_size = 4
    num_workers = 4
    gpu = -1

@ex.named_config
def mini():
    subset = 10

@ex.named_config
def plot():
    num_examples_to_plot = 10

########################################
########################################
# Load the Model
########################################
########################################
def load_the_model(weights):

    # load stored weights:
    if (weights is not None) and (os.path.exists(weights)):
         print("Loading weights file %s.." % weights)
         checkpoint = torch.load(weights, map_location='cpu')
         loss_ratio = checkpoint['args']['loss_ratio']
         batch_size_at_train = checkpoint['args']['batch_size']
         ds = load_dataset(checkpoint['args']['dataset'], include_trainset=False) # TODO: remove after installing the line below
         model = load_model(checkpoint['args']['model'], loss_ratio, num_internal_parts=ds.num_parts) # TODO: change to: num_internal_parts=ds_num_parts
         criterion = load_loss(checkpoint['args']['loss'], num_parts=ds.num_parts)
         print('Testing model {}'.format(model.__class__.__name__))
         if isinstance(criterion, tuple):
             print("Loss ratio is [{}}, {}}]" % (loss_ratio[0], loss_ratio[1]))
         print('Saved at epoch {}'.format(checkpoint['epoch']))
         model.load_state_dict(checkpoint['model_state_dict'])
         identify_model(checkpoint=checkpoint)
    else:
        sys.exit("Error: weights file does not exist")

    # set eval mode:
    model.eval()

    # set dirname, basename:
    dirname = os.path.dirname(weights)
    basename = os.path.splitext(os.path.basename(weights))[0]

    return model, basename, dirname, criterion, batch_size_at_train

def identify_model(filename=None, checkpoint=None):
    if filename is not None and os.path.exists(filename):
        print("loading file %s.." % filename)
        checkpoint = torch.load(filename)

    if checkpoint is not None:
        print('epoch is %d' % checkpoint['epoch'])
        if 'args' in checkpoint:
            print(checkpoint['args'])
        if 'current_train_loss' in checkpoint:
            print('current_train_loss = %.6f' % checkpoint['current_train_loss'])
        if 'current_val_loss' in checkpoint:
            print('current_val_loss = %.6f' % checkpoint['current_val_loss'])
        if 'best_val_metric' in checkpoint:
            print('best_val_metric = %.6f' % checkpoint['best_val_metric'])
        if 'hostname' in checkpoint:
            print('Host name: %s' % checkpoint['hostname'])
        if 'saved_to' in checkpoint:
            print('Saved to: %s' % checkpoint['saved_to'])
    else:
        print("path names or checkpoints do not exist..")

########################################
########################################
# Eval the model
########################################
########################################
def eval_model(total_predicted_class, total_gt_class, total_filenames, total_predicted_interp=None, total_gt_interp=None, outname='./'):
    # analyze results
    is_correct = np.array(total_predicted_class, dtype=int) == np.array(total_gt_class, dtype=int)
    total_test_accuracy = sum(is_correct) / len(total_filenames)
    incorrect_indices = np.where(is_correct == 0)[0]
    incorrect_files = [total_filenames[i] for i in incorrect_indices]
    incorrect_files_pred_classes = [total_predicted_class[i] for i in incorrect_indices]
    incorrect_files_gt_classes = [total_gt_class[i] for i in incorrect_indices]
    print("Number of incorrect detections is %d out of %d. Classification accuracy is.. %.6f"
          % (len(incorrect_files), len(total_filenames), total_test_accuracy))

    # plot and save class:
    if outname is not None:
        # plot
        out_grid = plot_misclassifications_grid(incorrect_files, incorrect_files_pred_classes)
        grid_name = outname + "_misclassifications.png"
        out_grid.save(grid_name)
        print("misclassified images were saved as %s" % grid_name)

        # save
        textfilename = outname + "_misclassifications.txt"
        with open(textfilename, 'w') as f:
            for indx, item in enumerate(incorrect_files):
                f.write("%s,%d,%d\n" % (item, incorrect_files_gt_classes[indx], incorrect_files_pred_classes[indx]))
        print("misclassified files were written to %s" % textfilename)

        # plot interps:
        if total_gt_interp is not None:
            fig_maps = plt.figure(figsize=(16, 10))
            fig_interps = plt.figure(figsize=(8, 8))
            if not os.path.exists(outname + '_plot_examples'):
                os.makedirs(outname + '_plot_examples')

            for k in range(min(len(total_gt_interp), len(total_filenames))):
                img = cv2.resize(cv2.imread(total_filenames[k]), (108, 108))
                img_basename = os.path.basename(total_filenames[k])
                print("plot example to %s .." % img_basename)
                fig_maps.clf()
                plot_maps(fig_maps, img, total_gt_interp[k], total_gt_class[k], total_predicted_class[k], total_predicted_interp[k])
                fig_maps.savefig(os.path.join(outname + '_plot_examples', 'maps_%s' % img_basename))
                fig_interps.clf()
                plot_maps(fig_interps, img, total_gt_interp[k], total_gt_class[k], total_predicted_class[k], total_predicted_interp[k], single_img_flag=True)
                fig_interps.savefig(os.path.join(outname + '_plot_examples', 'interp_%s' % img_basename))


########################################
########################################
# Run the model:
########################################
########################################
def run_model(model, basename, criterion, dataset, batch_size, subset, num_examples_to_plot, outname, device, num_workers):

    if model.output_type == 'img2class':
        num_examples_to_plot = 0

    # LOAD train data:
    ds = load_dataset(dataset, include_trainset=False)
    testsets = [[ds.testset, 'NaiveNegs-val'], [ds.testvalset, 'HardnegsDNN-test'], [ds.testvalset2, 'HardnegsInterp-test2']]

    textfilename = os.path.join(outname, '{}_evalReport.txt'.format(basename))
    with open(textfilename, 'w') as f:
        for testset in testsets:
            if subset is not None:
                testloader = torch.utils.data.DataLoader(testset[0], batch_size=batch_size,
                                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.randint(0, len(testset[0]), subset)),
                                                         shuffle=False, pin_memory=True, num_workers=num_workers)
            else:
                testloader = torch.utils.data.DataLoader(testset[0], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

            # run
            test_loss, test_interp_loss, test_class_loss, test_acc, test_auc, test_iou, test_misc = \
                run_epoch(epoch=0, dataset=testloader, optimizer=None, model=model, criterion=criterion, device=device, prefix='Testing', returned_examples=num_examples_to_plot)

            # eval
            test_outfilename = os.path.join(outname, '{}_{}'.format(basename, testset[1]))
            test_total_predicted_class, test_total_gt_class, test_total_filenames, test_total_predicted_interp, test_total_gt_interp = \
                test_misc[0], test_misc[1], test_misc[2], test_misc[3], test_misc[4]
            eval_model(total_predicted_class=test_total_predicted_class, total_gt_class=test_total_gt_class, total_filenames=test_total_filenames,
                       total_predicted_interp=test_total_predicted_interp, total_gt_interp=test_total_gt_interp, outname=test_outfilename)

            # report
            strr = '============= {}: ============='.format(testset[1])
            f.write(strr+'\n'); print(strr)
            strr = 'Loss/%s: %.2f' % (testset[1], test_loss)
            f.write(strr+'\n'); print(strr)
            strr = 'Loss/%s/interp: %.2f' % (testset[1], test_interp_loss)
            f.write(strr+'\n'); print(strr)
            strr = 'Loss/%s/class: %.2f' % (testset[1], test_class_loss)
            f.write(strr+'\n'); print(strr)
            strr = 'Accuracy/%s: %.2f' % (testset[1], test_acc)
            f.write(strr+'\n'); print(strr)
            strr = 'AUC/%s/class: %.2f' % (testset[1], test_auc)
            f.write(strr+'\n'); print(strr)
            strr = 'IOU/%s/class: %.2f' % (testset[1], np.mean(test_iou[1:]))
            f.write(strr+'\n'); print(strr)
            #strr = '======================================\n\n'
            strr = '\n'
            f.write(strr); print(strr)

    print("Evaluation report was saved to %s" % textfilename)
    return test_total_filenames, test_total_gt_class, test_total_predicted_class


########################################
########################################
# Main
########################################
########################################
@ex.automain
def main(_run):
    # INTRO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD model:
    model, basename, dirname, criterion, batch_size_at_train = load_the_model(_run.config['weights'])
    model.to(device)

    if _run.config['batch_size'] is None:
        batch_size = batch_size_at_train
    else:
        batch_size = _run.config['batch_size']

    outname = os.path.join(dirname, 'eval')
    if not os.path.exists(outname):
        os.makedirs(outname)

    basename = basename + "_testid_" + _run.config['testid']

    run_model(model, basename, criterion, _run.config['dataset'], batch_size, _run.config['subset'], _run.config['num_examples_to_plot'], outname, device, _run.config['num_workers'])

