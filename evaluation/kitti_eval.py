import sys
sys.path.append('../LIFE/')
sys.path.append('../LIFE/core')

import numpy as np
import torch
import argparse
from torch.utils.data import Dataset
import os
import cv2
from utils import frame_utils
from tqdm import tqdm
from flow_estimator import Flow_estimator
import warnings
warnings.filterwarnings('ignore')

class KITTI_Flow(Dataset):
    def __init__(self, path, type, version):
        self.pair_list = []
        self.type = type

        if version == '2015':
            image_folder = 'image_2'
        else:
            image_folder = 'image_0'

        image_list = sorted(os.listdir(os.path.join(path, image_folder)))
        flow_list = sorted(os.listdir(os.path.join(path, 'flow_%s' % type)))

        for idx in range(len(image_list)//2):
            im1_path = os.path.join(path, image_folder, image_list[idx * 2])
            im2_path = os.path.join(path, image_folder, image_list[idx * 2 + 1])
            flow_path = os.path.join(path, 'flow_%s' % type, flow_list[idx])

            self.pair_list += [{'id': idx,
                                'im1': im1_path,
                                'im2': im2_path,
                                'flow': flow_path}]

    def __getitem__(self, item):

        pair = self.pair_list[item]
        im1 = cv2.imread(pair['im1'])
        im2 = cv2.imread(pair['im2'])
        flow_gt, valid_gt = frame_utils.readFlowKITTI(pair['flow'])
        return im1, im2, flow_gt, valid_gt

    def __len__(self):

        return len(self.pair_list)


def evaluate_flow(dataloader):

    out_list, epe_list, fl_list = [], [], []

    for im1, im2, flow_gt, valid_gt in tqdm(dataloader):

        flow_gt = torch.from_numpy(flow_gt).cuda()
        valid_gt = torch.from_numpy(valid_gt).cuda()

        flow_pr = estimator.estimate(im1, im2)[0]

        epe = torch.norm(flow_pr.permute([1, 2, 0]) - flow_gt, dim=2)
        mag = torch.norm(flow_gt, dim=2)

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        fl_list.append(out[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    out_list = np.concatenate(out_list)
    epe = np.mean(epe_list)
    fl = 100 * np.mean(out_list)

    return {'epe': epe,
            'epe_list': epe_list,
            'fl': fl,
            'fl_list': fl_list}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation code')

    parser.add_argument('--iters',     type=int, default=12)
    parser.add_argument('--mixed_precision', type=bool, default=True)
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--data_dir',type=str, help='path to KITTI flow dataset')
    parser.add_argument('--model',   type=str,help='path to the test model')
    parser.add_argument('--version', type=str, default='2015', choices=['2015', '2012'])
    parser.add_argument('--gpus',    type=str, default='0')
    parser.add_argument('--type',    type=str, default='occ', choices=['occ', 'noc'])

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    estimator = Flow_estimator(args)
    dataloader = KITTI_Flow(args.data_dir, args.type, args.version)
    res = evaluate_flow(dataloader)

    print(os.path.basename(args.model), args.version, args.type)
    print('epe: {:.2f}, fl: {:.2f}'.format(res['epe'], res['fl']))
