"""
Fine-tuning script for downstream tasks.
"""

import torch
import os
import argparse
import numpy as np
import sys
import logging
import pandas as pd
import SimpleITK as sitk
from utils.parse_config import parse_config, set_random
from utils.dataloader import niiDataset
from network.unet import UNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from ruamel.yaml import YAML
from scipy import ndimage
from medpy import metric


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='mms',
                    help='name of training datasets')
parser.add_argument('--channel_num', type=int, default=1,
                    help='input channel of the network')
parser.add_argument('--num_classes', type=int, default=1,
                    help='output channel of network')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--subsets', nargs='+', default=['subset10', 'subset20'],
                    help='subsets used for fine-tuning')
parser.add_argument('--seg_class_num', type=int, default='4',
                    help='segmentation classes for downstream tasks')
args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_eval(predict, label, num_classes):
    # Computer Dice coefficient
    dice = np.zeros(num_classes)
    eps = 1e-7
    for c in range(num_classes):
        inter = 2.0 * (np.sum((predict == c) * (label == c), dtype=np.float32))
        p_sum = np.sum(predict == c, dtype=np.float32)
        gt_sum = np.sum(label == c, dtype=np.float32)
        dice[c] = (inter + eps) / (p_sum + gt_sum + eps)
    return dice[1:]


def assd_eval(predict,label,num_classes):
    assd_all = np.zeros(num_classes)
    for c in range(num_classes):
        reference = (label==c) * 1
        result = (predict==c) * 1
        sum_output = np.sum(result)
        if sum_output == 0:
            for c_new in range(num_classes):
                assd_all[c_new] = np.nan
        else:
            assd_all[c] = metric.binary.assd(result,reference)

    return assd_all[1:]


def test(config, ft_model, test_loader, list_data, subset, save_path):
    dataset = config['train']['dataset']
    if dataset == 'mms':
        num_classes = config['network']['n_classes_mms']
        log = pd.DataFrame(index=[], columns=['name', 'one_case_dice[0]', 'one_case_dice[1]', 'one_case_dice[2]',
                                              'one_case_assd[0]', 'one_case_assd[1]', 'one_case_assd[2]'])
    elif dataset == 'fb':
        num_classes = config['network']['n_classes_fb']
        log = pd.DataFrame(index=[], columns=['name', 'one_case_dice[0]', 'one_case_assd[0]'])

    device = torch.device('cuda:{}'.format(config['train']['gpu']))

    for data_loader in [test_loader]:
        all_batch_dice = []
        all_batch_assd = []
        with torch.no_grad():
            ft_model.eval()
            for it, (xt, xt_label, xt_name, lab_Img_dir) in enumerate(data_loader):
                xt = xt.to(device)
                xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                output = ft_model.test_(xt)
                output = output.squeeze(0)
                output = torch.argmax(output, dim=1)
                output = output.cpu().numpy()
                output_ = output
                output = output_.squeeze()
                if config['test']['save_result']:
                    lab_Img = sitk.ReadImage(lab_Img_dir[0])
                    lab_arr = sitk.GetArrayFromImage(lab_Img)
                    output_ = np.expand_dims(output_, axis=0)
                    if len(lab_arr.shape) == 4:
                        e, a, b, c = lab_arr.shape
                    elif len(lab_arr.shape) == 3:
                        e, b, c = lab_arr.shape
                    ee, aa, bb, cc = output_.shape
                    zoom = [1, 1, b/bb, c/cc]
                    output_ = ndimage.zoom(output_, zoom, order=0)
                    output_ = output_.squeeze(0).astype(np.float64)
                    name = str(xt_name)[2:-3]
                    results = save_path + '/nii/' + str(subset)
                    if not os.path.exists(results):
                        os.makedirs(results)
                    predict_dir = os.path.join(results, name)
                    out_lab_obj = sitk.GetImageFromArray(output_)
                    out_lab_obj.CopyInformation(lab_Img)
                    sitk.WriteImage(out_lab_obj, predict_dir)
                lab_Img = sitk.ReadImage(lab_Img_dir[0])
                lab_arr = sitk.GetArrayFromImage(lab_Img)
                if len(lab_arr.shape) == 4:
                    e, a, b, c = lab_arr.shape
                elif len(lab_arr.shape) == 3:
                    e, b, c = lab_arr.shape
                ee, bb, cc = output.shape
                zoom = [1, b / bb, c / cc]
                output = ndimage.zoom(output, zoom, order=0)
                xt_label = ndimage.zoom(xt_label, zoom, order=0)
                one_case_dice = dice_eval(output, xt_label, num_classes) * 100
                all_batch_dice += [one_case_dice]
                one_case_assd = assd_eval(output, xt_label, num_classes)
                all_batch_assd.append(one_case_assd)

                if dataset == 'mms':
                    tmp = pd.Series([
                        xt_name,
                        one_case_dice[0],
                        one_case_dice[1],
                        one_case_dice[2],
                        one_case_assd[0],
                        one_case_assd[1],
                        one_case_assd[2],
                    ], index=['name', 'one_case_dice[0]', 'one_case_dice[1]', 'one_case_dice[2]',
                              'one_case_assd[0]', 'one_case_assd[1]', 'one_case_assd[2]', ])
                if dataset == 'fb':
                    tmp = pd.Series([
                        xt_name,
                        one_case_dice[0],
                        one_case_assd[0]
                    ], index=['name', 'one_case_dice[0]', 'one_case_assd[0]'])
                log = log._append(tmp, ignore_index=True)

            if dataset == 'mms':
                all_batch_dice = np.array(all_batch_dice)
                all_batch_assd = np.array(all_batch_assd)
                all_batch_assd = all_batch_assd[~np.isnan(all_batch_assd).any(axis=1)]
                mean_dice = np.mean(all_batch_dice, axis=0)
                std_dice = np.std(all_batch_dice, axis=0)
                mean_assd = np.mean(all_batch_assd, axis=0)
                std_assd = np.std(all_batch_assd, axis=0)
                print(mean_dice, std_dice, mean_assd, std_assd)
            elif dataset == 'fb':
                all_batch_dice = np.array(all_batch_dice)
                all_batch_assd = np.array(all_batch_assd)
                non_nan_mask = ~np.isnan(all_batch_assd)
                all_batch_assd = all_batch_assd[non_nan_mask]
                mean_dice = np.mean(all_batch_dice, axis=0)
                std_dice = np.std(all_batch_dice, axis=0)
                mean_assd = np.mean(all_batch_assd, axis=0)
                std_assd = np.std(all_batch_assd, axis=0)
                print(mean_dice, std_dice, mean_assd, std_assd)

        model_dir = save_path + '/csv/' + str(subset)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log.to_csv(model_dir + '/' + str(subset) + '_log_test.csv', index=False)

        if dataset == 'mms':
            print('{}±{} {}±{} {}±{}'.format(np.round(mean_dice[0], 2), np.round(std_dice[0], 2),
                                             np.round(mean_dice[1], 2), np.round(std_dice[1], 2),
                                             np.round(mean_dice[2], 2), np.round(std_dice[2], 2)))
            print('{}±{}'.format(np.round(np.mean(mean_dice, axis=0), 2), np.round(np.mean(std_dice, axis=0), 2)))
            list_data.append('{}±{} {}±{} {}±{}'.format(np.round(mean_dice[0], 2), np.round(std_dice[0], 2),
                                                        np.round(mean_dice[1], 2), np.round(std_dice[1], 2),
                                                        np.round(mean_dice[2], 2), np.round(std_dice[2], 2)))
            list_data.append(
                '{}±{}'.format(np.round(np.mean(mean_dice, axis=0), 2), np.round(np.mean(std_dice, axis=0), 2)))
        elif dataset == 'fb':
            print('{}±{}'.format(np.round(mean_dice[0], 2), np.round(std_dice[0], 2)))
            list_data.append('{}±{}'.format(np.round(mean_dice[0], 2), np.round(std_dice[0], 2)))

        if dataset == 'mms':
            print('{}±{} {}±{} {}±{}'.format(np.round(mean_assd[0], 2), np.round(std_assd[0], 2),
                                             np.round(mean_assd[1], 2), np.round(std_assd[1], 2),
                                             np.round(mean_assd[2], 2), np.round(std_assd[2], 2)))
            print('{}±{}'.format(np.round(np.mean(mean_assd, axis=0), 2), np.round(np.mean(std_assd, axis=0), 2)))
            list_data.append('{}±{} {}±{} {}±{}'.format(np.round(mean_assd[0], 2), np.round(std_assd[0], 2),
                                                        np.round(mean_assd[1], 2), np.round(std_assd[1], 2),
                                                        np.round(mean_assd[2], 2), np.round(std_assd[2], 2)))
            list_data.append(
                '{}±{}'.format(np.round(np.mean(mean_assd, axis=0), 2), np.round(np.mean(std_assd, axis=0), 2)))
        elif dataset == 'fb':
            print('{}±{}'.format(np.round(mean_assd, 2), np.round(std_assd, 2)))
            list_data.append('{}±{}'.format(np.round(mean_assd, 2), np.round(std_assd, 2)))

    return list_data


def get_data_loader(config, dataset, subset):
    batch_size = config['train']['batch_size']
    data_root_mms = config['train']['data_root_mms']
    data_root_fb = config['train']['data_root_fb']

    if dataset == 'mms':
        train_img = data_root_mms + '/train/img/{}'.format(subset)
        train_lab = data_root_mms + '/train/lab/{}'.format(subset)
        valid_img = data_root_mms + '/valid/img/{}'.format(subset)
        valid_lab = data_root_mms + '/valid/lab/{}'.format(subset)
        test_img = data_root_mms + '/test/img/{}'.format(subset)
        test_lab = data_root_mms + '/test/lab/{}'.format(subset)
        train_dataset = niiDataset(train_img, train_lab, dataset=dataset, phase='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataset = niiDataset(valid_img, valid_lab, dataset=dataset, phase='valid')
        valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_dataset = niiDataset(test_img, test_lab, dataset=dataset, phase='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    elif dataset == 'fb':
        train_img = data_root_fb + '/{}/train/image'.format(subset)
        train_lab = data_root_fb + '/{}/train/label'.format(subset)
        valid_img = data_root_fb + '/{}/valid/image'.format(subset)
        valid_lab = data_root_fb + '/{}/valid/label'.format(subset)
        test_img = data_root_fb + '/{}/test/image'.format(subset)
        test_lab = data_root_fb + '/{}/test/label'.format(subset)
        train_dataset = niiDataset(train_img, train_lab, dataset=dataset, phase='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataset = niiDataset(valid_img, valid_lab, dataset=dataset, phase='valid')
        valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_dataset = niiDataset(test_img, test_lab, dataset=dataset, phase='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader


def train(config, train_loader, valid_loader, test_loader, subset, list_data, save_path):
    log_path = os.path.join(save_path, 'train_log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, f'{subset}.log'), level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    directory_path = save_path + '/txt/' + str(subset)
    file_path = os.path.join(directory_path, f'{subset}.txt')
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(file_path, 'w') as file:
        file.write('1' + "\n")

    dataset = config['train']['dataset']
    if dataset == 'mms':
        num_classes = config['network']['n_classes_mms']
    elif dataset == 'fb':
        num_classes = config['network']['n_classes_fb']

    # load pre_train model
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    ft_model = UNet(in_chns=args.channel_num, class_num=args.seg_class_num, pre_train=False).cuda()
    ft_model.train()

    checkpoint = torch.load("./models/mms/iter_40000.pth", map_location="cpu")
    model_dict = ft_model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    ft_model.load_state_dict(model_dict)

    # load train details
    num_epochs = config['train']['num_epochs']
    valid_epochs = config['train']['valid_epoch']
    best_dice = 0.
    best_epoch_num = 0
    trigger = 0
    for epoch in range(num_epochs):
        ft_model.train()
        seg_loss_epoch = AverageMeter()
        current_loss = 0.
        for i, (B, B_label, B_name, lab_path) in tqdm(enumerate(train_loader)):
            B = B.to(device).detach()
            B_label = B_label.to(device).detach()
            loss_seg = ft_model.train_(B, B_label)
            seg_loss_epoch.update(loss_seg)
            current_loss += 0
        loss_mean = current_loss / (i + 1)
        logging.info('Epoch[%d/%d]-Lr: %.8f --> Train...[Domain:%s seg Loss = %.4f, Fore_Dice_loss = %.4f]' %
                     (epoch + 1, num_epochs, ft_model.aux_dec1_opt.param_groups[0]['lr'], str(subset),
                      seg_loss_epoch.avg, loss_mean))
        if (epoch + 1) % valid_epochs == 0:
            current_dice = 0.
            with torch.no_grad():
                ft_model.eval()
                for it, (xt, xt_label, xt_name, lab_Img) in tqdm(enumerate(valid_loader)):
                    xt = xt.to(device)
                    xt_label = xt_label.numpy().squeeze().astype(np.uint8)
                    output = ft_model.test_(xt)
                    output = output.squeeze(0)
                    output = torch.argmax(output, dim=1)
                    output_ = output.cpu().numpy()
                    output = output_.squeeze()
                    one_case_dice = dice_eval(output, xt_label, num_classes) * 100
                    one_case_dice = np.array(one_case_dice)
                    one_case_dice = np.mean(one_case_dice, axis=0)
                    current_dice += one_case_dice
            dice_mean = current_dice / (it + 1)
            logging.info('\n Epoch[%4d/%4d] --> Validation...[Dice Coef: %.4f]' % (epoch + 1, num_epochs, dice_mean))
            dice_mean = current_dice / (it + 1)
            if dice_mean > best_dice:
                model_dir = save_path + "/model/" + str(subset)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                best_epoch = '{}/model-{}-{}-{:.3f}.pth'.format(model_dir, 'best', str(epoch + 1), dice_mean)
                torch.save(ft_model.state_dict(), best_epoch)
                torch.save(ft_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))
                logging.info('\n Epoch[%4d/%4d] --> Dice improved from %.4f in epoch %4d to %.4f ' %
                             (epoch + 1, num_epochs, best_dice, best_epoch_num, dice_mean))
                best_dice = current_dice / (it + 1)
                best_epoch_num = epoch + 1

                trigger = 0
            else:
                trigger += 1
            if trigger >= 40:
                print("=> early stopping")
                break

        ft_model.update_lr()
    ft_model.load_state_dict(torch.load(best_epoch, map_location='cpu'), strict=False)
    ft_model.eval()
    list_data.append('validation result')
    test(config, ft_model, valid_loader, list_data, subset, save_path)
    list_data.append('test result')
    test(config, ft_model, test_loader, list_data, subset, save_path)

    return list_data

def main():
    save_path = "./results/{}".format(args.datasets)
    now = datetime.now()
    save_path = os.path.join(save_path, now.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/config.cfg",
                        help='Path to the configuration file')
    args1 = parser.parse_args()

    config = args1.config
    config = parse_config(config)
    list_data = []
    print(config)

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    for dataset in ['mms']:
        for subset in args.subsets:
            print(dataset, subset)
            list_data.append(dataset)
            list_data.append(subset)
            train_loader, valid_loader, test_loader = get_data_loader(config, dataset, subset)
            list_data = train(config, train_loader, valid_loader, test_loader, subset, list_data, save_path)
            directory_path = save_path + '/txt/' + str(subset)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            file_path = os.path.join(directory_path, f'{subset}.txt')
            with open(file_path, 'w') as file:
                for line in list_data:
                    file.write(line + "\n")


if __name__ == '__main__':
    set_random()
    main()