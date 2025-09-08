"""
Script for self-supervised pretraining on the full training dataset.
"""

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import losses
from network.unet import UNet
from utils.dataloader import Pretrain_Datasets


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='default',
                    help='name of the experiment')
parser.add_argument('--datasets', type=str, default='mms',
                    help='name of training datasets')
parser.add_argument('--channel_num', type=int, default=1,
                    help='input channel of the network')
parser.add_argument('--mask_ratio', type=int, default=0.7,
                    help='mask ratio')
parser.add_argument('--alpha', type=float, default=0.5, help='loss weight')
parser.add_argument('--max_iterations', type=int, default=40000,
                    help='maximum number of training iterations')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--root_path', type=str, default=f"./outputs/...",
                    help='root path of data')
parser.add_argument('--sam_seg_class', type=int, default=10,
                    help='number of new_masks classes obtained from pre_process.py')
parser.add_argument('--recon_class_num', type=int, default=1,
                    help='reconstruction class num in pretrain')

args = parser.parse_args()


def get_h5_datasets(dataset):
    print(dataset)
    h5_data = []
    if "mms" in dataset:
        h5_data.append('mms_pre_processed_masks')
    if "fb" in dataset:
        h5_data.append('fb_pre_processed_masks')
    return h5_data


def contrastive_loss(anchor_embedding, positive_embedding, negative_embeddings, temperature=0.1):
    """
    Compute InfoNCE loss
    """
    pos_sim = F.cosine_similarity(anchor_embedding.unsqueeze(0), positive_embedding.unsqueeze(0), dim=1) / temperature
    neg_sim = F.cosine_similarity(anchor_embedding.unsqueeze(0), negative_embeddings, dim=1) / temperature

    pos_sim = pos_sim.squeeze(0)
    pos_exp = torch.exp(pos_sim)
    neg_exp = torch.exp(neg_sim)
    neg_sim_sum = torch.sum(neg_exp, dim=0)

    numerator = pos_exp
    denominator = pos_exp + neg_sim_sum
    loss = -torch.log(numerator / denominator)

    return loss.mean()


def compute_contrastive_loss(A_embeddings, A1_embeddings):
    """
    Compute contrastive loss over the entire batch.

    Args:
        A_embeddings: Feature embeddings of original images
        A1_embeddings: Feature embeddings of fused images
    """
    batch_size = A_embeddings.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        anchor_embedding = A_embeddings[i]
        positive_embedding = A1_embeddings[i]
        negative_indices = [j for j in range(batch_size) if j != i]
        negative_embeddings = A1_embeddings[negative_indices]

        loss = contrastive_loss(anchor_embedding, positive_embedding, negative_embeddings)
        total_loss += loss

    contrastive_loss_ = total_loss / batch_size

    return contrastive_loss_


def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    alpha = args.alpha
    mask_num = round(args.sam_seg_class * args.mask_ratio)
    remain_num = args.sam_seg_class - mask_num

    model = UNet(in_chns=args.channel_num, num_class_seg=args.sam_seg_class, num_class_recon=args.recon_class_num,
                 pre_train=True).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    h5_datasets = get_h5_datasets(args.datasets)
    train_set = Pretrain_Datasets(base_dir=args.root_path, datasets=h5_datasets, transform=None)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    dice_loss = losses.DiceLoss(args.sam_seg_class)
    mse_loss = nn.MSELoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            label_batch, name_batch = sampled_batch['label'], sampled_batch['name']
            mask_gt = sampled_batch['mask_gt'].cuda()
            label_batch = label_batch.cuda()

            unique_regions = torch.unique(mask_gt)
            num_regions = len(unique_regions)
            selected_regions = unique_regions[torch.randperm(num_regions)[:remain_num]]
            region_mask = torch.isin(mask_gt, selected_regions).unsqueeze(1)

            mask_gt = mask_gt - 1
            mask_gt = mask_gt.unsqueeze(1)
            masked_image = torch.full_like(label_batch, -2)
            masked_image[region_mask] = label_batch[region_mask]

            batch_size = label_batch.shape[0]
            random_indices = torch.randint(0, batch_size, (batch_size,), device=label_batch.device)
            background_mask = (masked_image == -2)
            masked_image[background_mask] = label_batch[random_indices][:, :, :, :][background_mask]

            A = label_batch
            A1 = masked_image
            A1_embeddings, recon_img, seg_logits = model(A1)
            _, A_embeddings = model.encoder(A)

            # calculate training loss
            fcl_loss = compute_contrastive_loss(A_embeddings, A1_embeddings)
            recon_outputs = torch.tanh(recon_img)
            ifr_loss = mse_loss(recon_outputs, label_batch)
            skd_loss = dice_loss(inputs=seg_logits, target=mask_gt, onehot=True, softmax=True)
            loss = skd_loss + ifr_loss + alpha * fcl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            tqdm.write('iteration %d : loss : %.4f Dice loss: %.4f' %
                       (iter_num, loss.item(), skd_loss.item()))

            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./models/{}/{}".format(args.datasets, args.exp_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
