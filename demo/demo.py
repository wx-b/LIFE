import sys
import os
import os.path as osp
os.chdir('..')
sys.path.append('.')
sys.path.append('./LIFE')
sys.path.append('./LIFE/core')
from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
from flow_estimator import Flow_estimator
from config import get_demo_args
from core.utils.utils import image_flow_warp
from tqdm import tqdm

def blend(im1, im2, im3):
    H, W = 480, 640
    ori_H, ori_W = scene_im.shape[:2]
    im1 = cv2.resize(im1, (W, H))
    im2 = cv2.resize(im2, (W, H))
    im3 = cv2.resize(im3, (W, H))
    flow = estimator.estimate(im1, im2)
    out = image_flow_warp(im3, flow[0].permute([1,2,0]),padding_mode='zeros')
    kernel = np.ones((3, 3), np.uint8)
    out = cv2.dilate(out, kernel, iterations=1).astype(np.uint8)
    intensity = np.linalg.norm(out, axis=2)
    mask = (intensity == 0)[:,:,np.newaxis]
    out = image_flow_warp(im3, flow[0].permute([1,2,0]),padding_mode='border')
    blend = (out * (1 - mask) + im1 * mask).astype(np.uint8)
    blend = cv2.resize(blend, (ori_W, ori_H))
    return blend

args = get_demo_args()
estimator = Flow_estimator(args)

movie_start_idx = args.movie_start_idx
video_start_idx = args.video_start_idx
frame_number = args.frame_number

data_path = args.data_path
vpath = osp.join(data_path, "video")
mpath = osp.join(data_path, "movie")
output_path = osp.join(data_path, args.output)
os.system("mkdir -p {}".format(output_path))

target_im = cv2.imread('./assets/imgs/the_starry_night.png')

for idx in tqdm(range(frame_number)):
    mid = movie_start_idx + idx
    vid = video_start_idx + idx

    scene_im  = cv2.imread(osp.join(vpath, "{:06d}.png".format(vid)))
    source_im = cv2.imread(osp.join(mpath, "{:04d}.png".format(mid)))
    out = blend(scene_im, target_im, source_im)

    cv2.imwrite(osp.join(output_path, "{:06d}.png".format(vid)), out)