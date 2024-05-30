import os
import torch

import matplotlib.pyplot as plt

from measure import compute_PSNR


def save_model(result_path, epoch, model, optimizer, file_name):
    best_model = os.path.join(result_path, "{}.ckpt".format(file_name))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, best_model)


def load_model(model, bestmodel_path):
    f = os.path.join(bestmodel_path, 'best_model.ckpt')
    checkpoint = torch.load(f)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)


def denormalize_(image, norm_range_max, norm_range_min):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    
    return image


def trunc(mat, trunc_max, trunc_min):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    
    return mat


def save_fig(x, y, pred, fig_name, original_result, pred_result, trunc_min, trunc_max, result_path):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0], original_result[1], original_result[2]), fontsize=20)

        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0], pred_result[1], pred_result[2]), fontsize=20)

        ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(result_path, 'result_{}.png'.format(fig_name)))
        plt.close()


def save_validation_fig(input, target, pred, data_range, psnr, trunc_min, trunc_max, result_path, epoch):
     # save validation fig
    input, target, pred = input.numpy(), target.numpy(), pred.numpy()
    quarter_psnr = compute_PSNR(input, target, data_range)

    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(input, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}".format(quarter_psnr))

    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}".format(psnr))

    ax[2].imshow(target, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(result_path, 'validation_{}.png'.format(epoch)))
    plt.close()
