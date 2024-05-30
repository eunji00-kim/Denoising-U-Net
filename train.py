import argparse
import time
import os
import torch

import numpy as np
import torch.nn as nn

from torch.backends import cudnn
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from models import UNet, UUNet
from dataloader import ct_dataset
from utils import denormalize_, trunc, save_validation_fig, save_model
from measure import compute_PSNR


def get_args():
    parser = argparse.ArgumentParser(description="Train Denoising U-Net")
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--train_path', type=str, default='train_dataset')
    parser.add_argument('--val_path', type=str, default='validation_dataset')
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--save_iters', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--min_range', type=float, default=-1024.0)
    parser.add_argument('--max_range', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--batch_size', type=str, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=str, default='cosineannealinglr')
    parser.add_argument('--t_max', type=str, default=100)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--augmentation', type=bool, default=False)
    
    return parser.parse_args()


def train(model, device, args):
    
    model = args.model
    train_path = args.train_path
    val_path = args.val_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_from = args.resume_from
    print_iters = args.print_iters
    save_iters = args.save_iters
    num_workers = args.num_workers
    min_range = args.min_range
    max_range = args.max_range
    trunc_min = args.trunc_min
    trunc_max = args.trunc_max
    weight_decay = args.weight_decay
    scheduler = args.scheduler
    t_max = args.t_max
    step_size = args.step_size
    gamma = args.gamma
    augmentation = args.augmentation
    result_path = args.result_path

    train_data = ct_dataset(train_path, min_range, max_range, augmentation=augmentation)
    val_data = ct_dataset(val_path, min_range, max_range, augmentation=False)

    train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataset = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    if scheduler == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
    else:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses = []
    val_losses = []
    val_psnrs = []
    count_iters = 0
    total_iters = (len(train_data) // batch_size) * epochs
    start_epoch = 1
    best_psnr = 0
    start_time = time.time()
    data_range = trunc_max - trunc_min

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print("Created a directory: {}".format(result_path))

    if resume_from:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, epochs+1):
        model.train()

        for iter_, (input, target) in enumerate(train_dataset):
            count_iters += 1

            input = input.float().to(device)
            target = target.float().to(device)

            pred = model(input).to(device)
            loss = criterion(pred, target)
            model.zero_grad()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # print
            if count_iters % print_iters == 0:
                print("EPOCH {} | ITER [{}/{}] | LOSS [{:.8f}] | TIME [{:.1f}s]".format(epoch, count_iters, total_iters, loss.item(), time.time()-start_time))
            
            if count_iters % save_iters == 0:
                save_model(result_path, epoch, model, optimizer, 'latest_model')
                np.save(os.path.join(result_path, 'loss_{}_iter.npy'.format(count_iters)), np.array(train_losses))

        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for val_iter_, (val_input, val_target) in enumerate(val_dataset):
                val_input = val_input.to(device)
                val_target = val_target.to(device)

                val_pred = model(val_input)
                val_loss = criterion(val_pred, val_target).item()

                # denormalize, truncate
                shape_ = val_input.shape[-1]
                
                for i in range(len(val_target)):
                    input, target, pred = val_input[i], val_target[i], val_pred[i]

                    input = trunc(denormalize_(input.view(shape_, shape_).cpu().detach(), max_range, min_range), trunc_max, trunc_min)
                    target = trunc(denormalize_(target.view(shape_, shape_).cpu().detach(), max_range, min_range), trunc_max, trunc_min)
                    pred = trunc(denormalize_(pred.view(shape_, shape_).cpu().detach(), max_range, min_range), trunc_max, trunc_min)

                    psnr = compute_PSNR(target, pred, data_range)
                    val_psnr += psnr
                
            val_loss /= len(val_data)
            val_psnr /= len(val_data)
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)

            print("Validation loss: {}".format(val_loss))
            print("PSNR: {}".format(val_psnr))

            if best_psnr < val_psnr:
                save_model(result_path, epoch, model, optimizer, 'best_model')
        
                best_psnr = val_psnr
        
        save_validation_fig(input, target, pred, data_range, psnr, trunc_min, trunc_max, result_path, epoch)


if __name__ == '__main__':

    cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)


    if args.model == 'unet':
        model = UNet().to(device)
    else:
        model = UUNet().to(device)

    train(model, device, args)
