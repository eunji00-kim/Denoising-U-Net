import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.backends import cudnn

from dataloader import ct_dataset
from models import UNet, UUNet
from measure import compute_measure
from utils import load_model, denormalize_, trunc, save_fig


def get_args():
    parser = argparse.ArgumentParser(description = "Test Denoising U-Net")

    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--data_path', type=str, default='npz_dataset/test_dataset')
    parser.add_argument('--min_range', type=float, default=-1024.0)
    parser.add_argument('--max_range', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--result_path', type=str, default='result1')

    return parser.parse_args()


def test(model, device, args):

    data_path = args.data_path
    min_range = args.min_range
    max_range = args.max_range
    trunc_min = args.trunc_min
    trunc_max = args.trunc_max
    result_fig = args.result_fig
    result_path = args.result_path

    del model
    model = UNet().to(device)

    load_model(model, result_path)
    model.eval()

    dataset = ct_dataset(data_path, min_range, max_range)
    test_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    with torch.no_grad():
        for iter_, (input, target) in enumerate(test_dataset):
            shape_ = input.shape[-1]
            input = input.float().to(device)
            target = target.float().to(device)

            pred = model(input)

            # denormalize, truncate
            input = trunc(denormalize_(input.view(shape_, shape_).cpu().detach(), max_range, min_range), trunc_max, trunc_min)
            target = trunc(denormalize_(target.view(shape_, shape_).cpu().detach(), max_range, min_range), trunc_max, trunc_min)
            pred = trunc(denormalize_(pred.view(shape_, shape_).cpu().detach(), max_range, min_range), trunc_max, trunc_min)

            data_range = trunc_max - trunc_min

            original_result, pred_result = compute_measure(input, target, pred, data_range)
            ori_psnr_avg += original_result[0]
            ori_ssim_avg += original_result[1]
            ori_rmse_avg += original_result[2]
            pred_psnr_avg += pred_result[0]
            pred_ssim_avg += pred_result[1]
            pred_rmse_avg += pred_result[2]

            # save result figure
            if result_fig:
                save_fig(input, target, pred, iter_, original_result, pred_result, trunc_min, trunc_max, result_path)

        print('\n')
        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(test_dataset), 
                                                                                            ori_ssim_avg/len(test_dataset), 
                                                                                            ori_rmse_avg/len(test_dataset)))
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(test_dataset), 
                                                                                                pred_ssim_avg/len(test_dataset), 
                                                                                                pred_rmse_avg/len(test_dataset)))
        
        with open(os.path.join(result_path, 'average_results.txt'), 'w') as avg_file:
            avg_file.write('Average Results:\n')
            avg_file.write('Original average results:\n')
            avg_file.write(f"PSNR: {ori_psnr_avg/len(test_dataset)}, SSIM: {ori_ssim_avg/len(test_dataset)}, RMSE: {ori_rmse_avg/len(test_dataset)}\n\n")
            avg_file.write('Predicted average results:\n')
            avg_file.write(f"PSNR: {pred_psnr_avg/len(test_dataset)}, SSIM: {pred_ssim_avg/len(test_dataset)}, RMSE: {pred_rmse_avg/len(test_dataset)}\n")


if __name__ == '__main__':
    cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'unet':
        model = UNet().to(device)
    else:
        model = UUNet().to(device)

    test(model, device, args)
