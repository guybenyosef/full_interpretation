import torch
import torch.optim as optim
import os
import socket
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sacred import Experiment
from sacred.observers import FileStorageObserver

import CONSTS
from all_models import load_model
from all_losses import load_loss
from datasets import load_dataset
from engine import sample_dataset, run_epoch
from utils.interpretation_utils import convert_onehot, plot_maps, compute_auc_multiclass, compute_iou_interpretation, better_hparams, plot_misclassifications_grid

ex = Experiment('SemanticSupport')
logs_dir = 'storage'
#logs_dir = 'tmp'

ex.observers.append(FileStorageObserver.create(logs_dir))
def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if 'it/s]' not in line)
    return '\n'.join(lines)
ex.captured_out_filter = remove_progress

@ex.config
def config():
    model = 'UNetMirc'  # 'choose between: 'ResNet', 'UNetMirc', 'RecUNetMirc', 'RecUNetMirc', 'RecUNetMircTD', 'RecUNetMircMulti'
    loss = 'WeightedCrossEnt'  # 'Loss function for training the model'. Could be also array of [interp_loss, class_loss]
    dataset = 'HorseHead'  # 'The name of train/test sets'
    neg_set = 1  # Set of negative examples in train: (0) Small (usually saved for val), Medium (1), Large (2)
    batch_size = 64  # 'Specify the number of batches'
    num_workers = 32  # 'Specify the number of cpu cores'
    epochs = 100  # 'Specify the number of epochs to train'
    optimizer = 'ADAM'  # 'Optimizer (default: ADAM)'
    learningrate = 1e-4  # 'learning rate (default: 0.001)'
    loss_ratio = [1.0, 1.0]  # [interp, class]
    subset = 10000  # the subset size of negative examples added to each epoch. 'None' means adding all negative examples
    evaluation_interval = 1 # interval evaluations on validation set
    overfit = False
    plotflag = False
    data_aug = False
    seed = 1234

@ex.named_config
def overfit():
    neg_set = 0
    subset = None
    overfit = True
    batch_size = 1
    num_workers = 1
    epochs = 5000
    evaluation_interval = 10

# -- DATASETS: --
@ex.named_config
def mis():
    dataset = 'ManInSuit'

@ex.named_config
def eye():
    dataset = 'HumanEye'

# -- MODELS: --
@ex.named_config
def vanilla():
    model = 'ResNet18'
    loss = 'WeightedCrossEnt'
    loss_ratio = [0.0, 1.0]
    batch_size = 200
    subset = 1000
    epochs = 100

@ex.named_config
def resnet50():
    model = 'ResNet50'

@ex.named_config
def interp():
    model = 'UNetMirc'
    loss = 'CrossEntArray'  #'CrossEnt'
    loss_ratio = [1.0, 0.0]
    batch_size = 64
    neg_set = 0  # not used and therefore take the smallest test to cut costs
    subset = 0
    epochs = 1000
    evaluation_interval = 10

@ex.named_config
def deeplab():
    model = 'DeepLabMirc'

@ex.named_config
def fcn():
    model = 'FCNMirc'

@ex.named_config
def dual():
    model = 'RecUNetMirc'
    loss = ['CrossEntArray', 'WeightedCrossEntReduced'] #'WeightedCrossEnt']  #['CrossEntArray', 'CrossEnt'] # [interp, class]
    loss_ratio = [1.0, 1.0]  #[interp, class]
    batch_size = 64
    subset = 1000
    epochs = 100

@ex.named_config
def dualtd():
    model = 'RecUNetMircTD'
    loss = ['CrossEntArray', 'WeightedCrossEntReduced']  # [interp, class]
    loss_ratio = [1.0, 1.0]  # [interp, class]
    batch_size = 64
    subset = 1000
    epochs = 100

@ex.named_config
def dualmulti():
    model = 'RecUNetMircMulti'
    loss = ['CrossEntArray', 'WeightedCrossEntReduced']  # [interp, class]
    loss_ratio = [1.0, 1.0]  # [interp, class]
    batch_size = 16
    subset = 1000
    epochs = 100

# -- OPTIONS: --
@ex.named_config
def long():
    epochs = 1000
    evaluation_interval = 10

@ex.named_config
def verylong():
    epochs = 5000
    evaluation_interval = 10

@ex.named_config
def veryverylong():
    epochs = 10000
    evaluation_interval = 100

# boost interp
@ex.named_config
def boost():
    loss_ratio = [10.0, 1.0]  # [interp, class]

# boost interp more
@ex.named_config
def superboost():
    loss_ratio = [100.0, 1.0]  # [interp, class]

# ~10K set of negative examples
@ex.named_config
def small():
    neg_set = 0

# ~100K set of negative examples
@ex.named_config
def medium():
    neg_set = 1

# ~10M set of negative examples
@ex.named_config
def large():
    neg_set = 2

# reduce the classification stream in a dual model
@ex.named_config
def noclass():
    subset = 0
    neg_set = 0  # not used and therefore take the smallest test to cut costs
    loss_ratio = [1.0, 0.0]  # [interp, class]
    epochs = 1000
    evaluation_interval = 10

# reduce the interpretation stream in a dual model
@ex.named_config
def nointerp():
    loss_ratio = [0.0, 1.0]  # [interp, class]

# On each batch put a set of negative examples in equal size set of positive
@ex.named_config
def equal():
    subset = 130
    epochs = 1000
    evaluation_interval = 10

# Use Dice loss (TODO Dice -- still not working well)
@ex.named_config
def dice():
    loss = 'Dice' #['Dice', 'WeightedCrossEnt']

@ex.named_config
def dicedual():
    loss = ['Dice', 'WeightedCrossEnt']

# Use data augmentation for positive examples
@ex.named_config
def aug():
    data_aug = True

# low-power parameters suitable to a cpu machine
@ex.named_config
def mac():
    num_workers = 4
    batch_size = 4
    epochs = 10

########################################
########################################
# Utils
########################################
########################################
def save_model(filename, epoch, model, args, current_train_loss, current_val_loss, best_val_metric, num_parts, hostname):
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'epoch': epoch,
        'best_val_metric': best_val_metric,
        'current_train_loss': current_train_loss,
        'current_val_loss': current_val_loss,
        'hostname': hostname,
        'ds_num_parts': num_parts,
        'saved_to': filename
    }, filename)
    print("model saved to {}".format(filename))


########################################
########################################
# Main
########################################
########################################
@ex.automain
def main(_run):
    print(_run.config)
    torch.manual_seed(_run.config['seed'])
    np.random.seed(_run.config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device.type != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    hostname = socket.getfqdn()

    # LOAD train data:
    ds = load_dataset(_run.config['dataset'], _run.config['neg_set'], data_aug=_run.config['data_aug'])
    trainloader, valloader, testloader = sample_dataset(ds.trainset, ds.testset, ds.testvalset2,
                                             _run.config['overfit'], _run.config['subset'], _run.config['batch_size'], _run.config['num_workers'], verbose=True)

    # LOAD model:
    model = load_model(_run.config['model'], _run.config['loss_ratio'], num_internal_parts=ds.num_parts)
    if device.type != 'cpu':
        model.to(device)
    print('training model %s..' % model.__class__.__name__)
    basename = _run.config['dataset'] + '_' + _run.config['model']

    # loss
    class_weights = [1/f for f in ds.trainset.class_ratio]
    class_weights = None  #torch.Tensor(class_weights).to(device)
    criterion = load_loss(_run.config['loss'], num_parts=ds.num_parts, class_weights=class_weights, device=device)
    if isinstance(_run.config['loss'], tuple):
        loss_ratio = _run.config['loss_ratio']
        print("Loss ratio is [{}}, {}}]" % (loss_ratio[0], loss_ratio[1]))

    # Optimizer:
    if _run.config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=_run.config['learningrate'], momentum=0.99)
    elif _run.config['optimizer'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=_run.config['learningrate'], betas=(0.9, 0.999), weight_decay=1e-5)
    else:
        sys.exit('ERROR: optimizer name does not exist..')

    # logs:
    log_folder = os.path.join(logs_dir, 'logs', _run.config['dataset'], _run.config['model'], '{}'.format(_run.config['loss_ratio']), _run._id)
    writer = SummaryWriter(log_dir=log_folder)
    metric_dict = {'AUC/Best_Validation': 0, 'IOU/Best_Validation': 1, 'AUC/Test': 2}  # 'Loss/Validation': 3,
    run_dict = {k: v.__repr__() if isinstance(v, list) else v for k, v in _run.config.items() if v is not None}
    sei = better_hparams(writer, hparam_dict=run_dict, metric_dict=metric_dict)
    if _run.config['plotflag']:
        fig = plt.figure(figsize=(16, 10))

    print("Training in batches of size %d.." % _run.config['batch_size'])
    print('Training on machine name %s..' % hostname)
    print("log dir is: {}".format(log_folder))
    print("Using %s optimizer, lr=%f.." % (_run.config['optimizer'], _run.config['learningrate']))

    # Train & Evaluate:
    best_val_metric_class = 0.0
    best_val_metric_interp = 0.0
    with tqdm(total=_run.config['epochs']) as pbar_main:
        for epoch in range(1, _run.config['epochs']+1):
            pbar_main.update()

            if _run.config['subset'] is not None:  # update samples from train set
                trainloader, _, _ = sample_dataset(ds.trainset, ds.testset, ds.testvalset2,
                                                   _run.config['overfit'], _run.config['subset'], _run.config['batch_size'], _run.config['num_workers'])

            train_loss, train_interp_loss, train_class_loss, train_acc, train_auc, train_iou, train_misc = \
                run_epoch(epoch, dataset=trainloader, optimizer=optimizer, model=model, criterion=criterion, device=device, prefix='Training')

            if epoch == 1 or epoch % _run.config['evaluation_interval'] == 0:
                val_loss, val_interp_loss, val_class_loss, val_acc, val_auc, val_iou, val_misc = \
                    run_epoch(epoch, dataset=valloader, optimizer=None, model=model, criterion=criterion, device=device, prefix='Validating')

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Train/interp', train_interp_loss, epoch)
            writer.add_scalar('Loss/Train/class', train_class_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Loss/Val_interp', val_interp_loss, epoch)
            writer.add_scalar('Loss/Val_class', val_class_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('AUC/Train', train_auc, epoch)
            writer.add_scalar('AUC/Validation', val_auc, epoch)
            writer.add_scalar('IOU/Train_all', np.mean(train_iou), epoch)
            writer.add_scalar('IOU/Val_all', np.mean(val_iou), epoch)
            for cls_indx in range(len(train_iou)):
                writer.add_scalar('IOU/Train_part%d' % (cls_indx+1), train_iou[cls_indx], epoch)
                writer.add_scalar('IOU/Val_part%d' % (cls_indx+1), val_iou[cls_indx], epoch)


            # Plot results:
            # =================
            if _run.config['plotflag'] and model.output_type != 'img2class':
                train_imgs, train_interps, train_labels, train_interp_out, train_class_out, train_filenames = \
                train_misc[0], train_misc[1], train_misc[2], train_misc[3], train_misc[4], train_misc[5]

                val_imgs, val_interps, val_labels, val_interp_out, val_class_out, val_filenames = val_misc[0], val_misc[
                    1], val_misc[2], val_misc[3], val_misc[4], val_misc[5]

                fig.clf()
                plot_maps(fig, train_imgs[0], train_interps[0], train_labels[0], train_class_out[0], train_interp_out[0])
                writer.add_figure(tag='Images/Train', figure=fig, global_step=epoch, close=False, walltime=None)

                fig.clf()
                plot_maps(fig, val_imgs[0], val_interps[0], val_labels[0], val_class_out[0], val_interp_out[0])
                writer.add_figure(tag='Images/Val', figure=fig, global_step=epoch, close=False, walltime=None)

            # Update best:
            # =================
            if val_auc > best_val_metric_class:
                filename = os.path.join(log_folder, 'weights_{}_best_class.pth'.format(basename))
                best_val_metric_class = val_auc
                save_model(filename, epoch, model, _run.config, train_loss, val_loss, best_val_metric_class, ds.num_parts, hostname)
                writer.add_scalar('AUC/Best_Validation', best_val_metric_class, epoch)
                test_loss, test_interp_loss, test_class_loss, test_acc, test_auc, test_iou, test_misc = \
                    run_epoch(epoch, dataset=testloader, optimizer=None, model=model, criterion=criterion, device=device, prefix='Testing')
                writer.add_scalar('AUC/Test', test_auc, epoch)

            if np.mean(val_iou) > best_val_metric_interp:
                filename = os.path.join(log_folder, 'weights_{}_best_interp.pth'.format(basename))
                best_val_metric_interp = np.mean(val_iou)
                save_model(filename, epoch, model, _run.config, train_loss, val_loss, best_val_metric_class, ds.num_parts, hostname)
                writer.add_scalar('IOU/Best_Validation', best_val_metric_interp, epoch)
                # test_loss, test_interp_loss, test_class_loss, test_acc, test_auc, test_iou, test_misc = \
                # run_epoch(epoch, dataset=testloader, optimizer=optimizer, model=model, criterion=criterion, device=device)
                # writer.add_scalar('IOU/Test', test_iou, epoch)


    print('Finished Training')

    # ===============
    # Save & Close:
    # ===============
    filename = os.path.join(log_folder, 'weights_{}_ep_{}.pth'.format(basename, epoch))
    save_model(filename, epoch, model, _run.config, train_loss, val_loss, best_val_metric_class, ds.num_parts, hostname)
    writer.file_writer.add_summary(sei)
    writer.close()  # close tensorboard
