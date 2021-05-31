import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm

#from utils import logger
from utils import data_set
from utils import data_vis
from utils import augmentation as aug
from utils import metrics
from models import unet

import torch
import torch.optim as optim
import time
import argparse
import shutil
import os


def main(data_path, batch_size, num_epochs, start_epoch, learning_rate, momentum, bands, logger_freq, run, resume,
        loss_sampling, loss_sample_k, num_workers=2):
    """

    Args:
        data_path:
        batch_size:
        num_epochs:

    Returns:

    """
    since = time.time()
    cur_time = since

    # get files directory (assuming index file is at root level in data dir)
    data_root_dir = data_path.split('/')[-2]

    # get model
    model = unet.UNetSmall(num_channels=len(bands)+2)
    #model = unet.UNet(num_channels=len(bands)+2)

    if torch.cuda.is_available():
        model = model.cuda()

    # set up binary cross entropy and dice loss
    #criterion = metrics.BCEDiceLoss()
    
    if loss_sampling:
        criterion = nn.BCELoss(reduction='none')
        #criterion = nn.L1Loss(reduction='none')
    else:
        criterion = nn.BCELoss()
        #criterion = nn.L1Loss()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # decay LR
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=18, gamma=0.1)

    # starting params
    best_loss = 999

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']

            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    train_dataset = data_set.WildfireSmokeDataset(data_path, root_dir=data_root_dir, train_val_test='train', 
                                     bands=bands, transform=transforms.Compose([aug.ToTensorTarget()]))

    val_dataset = data_set.WildfireSmokeDataset(data_path, root_dir=data_root_dir, train_val_test='valid', 
                                     bands=bands, transform=transforms.Compose([aug.ToTensorTarget()]))

    # creating loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=num_workers, shuffle=False)

    # loggers    
    train_logger = SummaryWriter(f'../logs/run_{run}/training')
    val_logger = SummaryWriter(f'../logs/run_{run}/validation')
    
    progress_logger(f"START TRAINING| run {run} | start lr: {lr_scheduler.get_last_lr()}| bs: {batch_size}| bands: {bands}",
                    log_fn_slug=f"../training_logs/run_{run}_training_log")
    
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}| lr: {lr_scheduler.get_last_lr()}')
        print('-' * 10)
        
        progress_logger(f'Epoch {epoch}/{num_epochs - 1}| lr: {lr_scheduler.get_last_lr()}', 
                        log_fn_slug=f"../training_logs/run_{run}_training_log")

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        train_metrics = train(train_dataloader, model, criterion, optimizer, lr_scheduler, train_logger, epoch, 
                              run, logger_freq, loss_sampling, loss_sample_k)
        valid_metrics = validation(val_dataloader, model, criterion, val_logger, epoch, 
                                   run, logger_freq, loss_sampling)

        # store best loss and save a model checkpoint
        is_best = valid_metrics['valid_loss'] < best_loss
        best_loss = min(valid_metrics['valid_loss'], best_loss)
        save_checkpoint({
            'epoch': epoch,
            'arch': 'UNetSmall',
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()}, 
            is_best,
            checkpoint_fn=f"../checkpoints/checkpoint_run{run}.pth.tar",
            best_fn=f"../checkpoints/best_run{run}.pth.tar")

        total_time = time.time() - since
        cur_elapsed = time.time() - cur_time
        cur_time = time.time()
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

        # log training progress
        progress_logger(f"Epoch {epoch}| " + 
                        f"train_loss: {train_metrics['train_loss']:.4f}| " + \
                        f"train_acc: {train_metrics['train_acc']:.4f}| " + \
                        f"train_iou: {train_metrics['train_iou']:.4f}| " + \
                        f"val_loss: {valid_metrics['valid_loss']:.4f}| " + \
                        f"val_acc: {valid_metrics['valid_acc']:.4f}| " + \
                        f"val_iou: {valid_metrics['valid_iou']:.4f}| " + \
                        f"best_model: {is_best}| " + \
                        f"cur_time: {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s| " + \
                        f"tot_time: {total_time // 60:.0f}m {total_time % 60:.0f}s",
               log_fn_slug=f"../training_logs/run_{run}_training_log")

        progress_logger(f'-'*10,log_fn_slug=f"../training_logs/run_{run}_training_log")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    progress_logger(f'Total elapsed time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s',
               log_fn_slug=f"../training_logs/run_{run}_training_log")


def train(train_loader, model, criterion, optimizer, scheduler, logger, epoch_num, run, 
          logger_freq=4, loss_sampling=False, loss_sample_k=1):
    """

    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:

    Returns:

    """
    # set model to train mode
    model.train()
    
    # logging accuracy and loss
    train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()
    train_iou = metrics.MetricTracker()

    log_iter = len(train_loader)//logger_freq

    scheduler.step()

    # iterate over data
    for idx, data in enumerate(tqdm(train_loader, desc="training")):

        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(data['sat_img'].cuda())
            labels = Variable(data['map_img'].cuda())
        else:
            inputs = Variable(data['sat_img'])
            labels = Variable(data['map_img'])

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        outputs = torch.nn.functional.sigmoid(outputs)

        # calculate loss
        loss = criterion(outputs, labels)
        
        if loss_sampling:
            # training loss
            loss = criterion(outputs, labels)

            N, C, H, W = loss.shape

            # sum loss over the H, W dimensions and take avg (like usual)
            loss = loss.sum(dim=(2,3), keepdims=True).div(H*W)

            # get the top k loss from batch and set to 0
            # so that these data points don't contribute to update
            topk_idx = torch.topk(loss, k=loss_sample_k, dim=0)[1]
            loss.scatter_(0, topk_idx, 0)
            
            # only avg loss over remaining low loss samples
            num_good_training_samples = N-loss_sample_k
            loss = loss.sum().div(num_good_training_samples)
                
        # backward
        loss.backward()
        optimizer.step()

        train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        train_loss.update(loss.item(), outputs.size(0))
        train_iou.update(metrics.get_iou(outputs, labels), outputs.size(0))
        
        if idx % 5 == 0:
            progress_logger(f"    epoch: {epoch_num}| " + \
                            f"batch: {idx}| " + \
                            f"train_loss: {train_loss.avg:.4f}| " + \
                            f"train_acc: {train_acc.avg:.4f}| " + \
                            f"train_iou: {train_iou.avg:.4f}",
                   log_fn_slug=f"../training_logs/run_{run}_training_log")
        
        # tensorboard logging
        if idx % log_iter == 0:

            step = (epoch_num*logger_freq)+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': train_loss.avg,
                'accuracy': train_acc.avg
            }

            for tag, value in info.items():
                logger.add_scalar(tag, value, step)

            # log weights, biases, and gradients
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), step, bins='auto')
                logger.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), step, bins='auto')

            # log the sample images
            log_img = data_vis.show_tensorboard_image(data['sat_img'], data['map_img'], outputs)
            logger.add_figure('train_images', log_img, step)
            

    print(f"  Training Loss: {train_loss.avg:.4f} Acc: {train_acc.avg:.4f} IoU: {train_iou.avg:.4f}")
    print()

    return {'train_loss': train_loss.avg, 'train_acc': train_acc.avg, 'train_iou': train_iou.avg}


def validation(valid_loader, model, criterion, logger, epoch_num, run, logger_freq=4, loss_sampling=False):
    """

    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:

    Returns:

    """
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()
    valid_iou = metrics.MetricTracker()

    log_iter = len(valid_loader)//logger_freq

    # switch to evaluate mode and turn off gradient tracking
    model.eval()
    with torch.no_grad():  
        
        # iterate over data
        for idx, data in enumerate(tqdm(valid_loader, desc='validation')):

            # get the inputs and wrap in Variable
            if torch.cuda.is_available():
                inputs = Variable(data['sat_img'].cuda(), volatile=True)
                labels = Variable(data['map_img'].cuda(), volatile=True)
            else:
                inputs = Variable(data['sat_img'], volatile=True)
                labels = Variable(data['map_img'], volatile=True)

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)

            # calculate loss
            loss = criterion(outputs, labels)
            
            if loss_sampling:
                loss = loss.mean()


            valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            valid_loss.update(loss.item(), outputs.size(0))
            valid_iou.update(metrics.get_iou(outputs, labels), outputs.size(0))

            if idx % 5 == 0:
                progress_logger(f"    epoch: {epoch_num}| " + \
                                f"batch: {idx}| " + \
                                f"val_loss: {valid_loss.avg:.4f}| " + \
                                f"val_acc: {valid_acc.avg:.4f}| " + \
                                f"val_iou: {valid_iou.avg:.4f}",
                       log_fn_slug=f"../training_logs/run_{run}_training_log")

            # tensorboard logging
            if idx % log_iter == 0:

                step = (epoch_num*logger_freq)+(idx/log_iter)

                # log accuracy and loss
                info = {
                    'loss': valid_loss.avg,
                    'accuracy': valid_acc.avg
                }

                for tag, value in info.items():
                    logger.add_scalar(tag, value, step)

                # log the sample images
                log_img = data_vis.show_tensorboard_image(data['sat_img'], data['map_img'], outputs)
                logger.add_figure('valid_images', log_img, step)

    print(f"  Validation Loss: {valid_loss.avg:.4f} Acc: {valid_acc.avg:.4f} IoU: {valid_iou.avg:.4f}")
    print()

    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg, 'valid_iou': valid_iou.avg}


# create a function to save the model state (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
def save_checkpoint(state, is_best, checkpoint_fn="../checkpoints/checkpoint.pth.tar", best_fn="../checkpoints/best_model.pth.tar"):
    """
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, checkpoint_fn)
    if is_best:
        shutil.copyfile(checkpoint_fn, best_fn)

## logger for training
def progress_logger(info_str, log_fn_slug="../training_logs/training_log"):
    ## write to log file
    log = open(log_fn_slug+'.txt', "a+")  # append mode and create file if it doesnt exist
    log.write(info_str +
              "\n")
    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wildfire Smoke Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset csv')
    parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='epoch to start from (used with resume flag')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--bands', nargs="+", default=["true_color"],
                        help='input channels to read in as input')
    parser.add_argument('--logger-freq', default=10, type=int, metavar='N',
                        help='number of times to log per epoch')
    parser.add_argument('--run', default=0, type=int, metavar='N',
                        help='number of run (for tensorboard logging)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--loss-sampling', action='store_true',
                        help='Set high loss training samples in a batch to 0')
    parser.add_argument('--loss-sample-k', default=1, type=int, metavar='N',
                        help='number of high loss samples to drop from gradient update')

    args = parser.parse_args()

    # check if there are previous runs
    experiments = [int(run_dir.split('_')[1]) for run_dir in os.listdir('../logs') if run_dir not in ['.DS_Store', 'old_runs']]
    if args.run not in experiments:
        run_num = args.run
    elif (len(experiments) > 0):
        run_num = max(experiments) + 1

    # run training
    main(args.data, batch_size=args.batch_size, num_epochs=args.epochs, start_epoch=args.start_epoch, learning_rate=args.lr,
         momentum=args.momentum, bands=args.bands, logger_freq=args.logger_freq, run=run_num, resume=args.resume, 
         loss_sampling=args.loss_sampling, loss_sample_k=args.loss_sample_k)
