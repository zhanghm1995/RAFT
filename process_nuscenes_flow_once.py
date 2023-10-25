'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-21 23:58:25
Email: haimingzhang@link.cuhk.edu.cn
Description: Process the nuscense sequence dataset once a time.
'''

"""
This script computes all pairwise RAFT optical flow fields
for each pair, we use previous flow as initialization to compute the current flow
"""

import sys

sys.path.append('core')

import argparse
import os
import os.path as osp
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils.utils import InputPadder
from utils import flow_viz
import cv2
import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cuda'


def load_image(imfile, resize=False, resize_dims=None):
    img_input = Image.open(imfile)
    if resize:
        assert resize_dims is not None
        img_input = img_input.resize(resize_dims)

    img = np.array(img_input).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flowpath, rfpath):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=1)

    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    cv2.imwrite(flowpath, flo)
    cv2.imwrite(rfpath, img_flo[:, :, [2,1,0]])


def load_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model


def run_exhaustive_flow(model, args, resize, resize_dims):
    data_dir = args.data_dir
    assert os.path.exists(data_dir)
    
    print('computing all pairwise optical flows for {}...'.format(data_dir))

    flow_out_dir = os.path.join(data_dir, 'raft_flow')
    os.makedirs(flow_out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(data_dir, 'color', '*')))
    num_imgs = len(img_files)
    print(f"There are {num_imgs} images in {data_dir}")
    pbar = tqdm(total=num_imgs * 10)

    neighbor_length = 5
    with torch.no_grad():
        for i in range(num_imgs - 1):
            flow_low_prev = None
            for j in range(i + 1, min(i + 1 + neighbor_length, num_imgs)):
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                image1 = load_image(imfile1, resize, resize_dims)
                image2 = load_image(imfile2, resize, resize_dims)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low_prev)
                flow_up = padder.unpad(flow_up)

                flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                save_file = os.path.join(flow_out_dir,
                                         '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
                np.save(save_file, flow_up_np)
                flow_low_prev = flow_low
                pbar.update(1)

        for i in range(num_imgs - 1, 0, -1):
            flow_low_prev = None
            for j in range(i - 1, max(i - 1 - neighbor_length, -1), -1):
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                image1 = load_image(imfile1, resize, resize_dims)
                image2 = load_image(imfile2, resize, resize_dims)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low_prev)
                flow_up = padder.unpad(flow_up)

                flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                save_file = os.path.join(flow_out_dir,
                                         '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
                np.save(save_file, flow_up_np)
                flow_low_prev = flow_low
                pbar.update(1)
        pbar.close()
        print('computing all pairwise optical flows for {} is done \n'.format(data_dir))


def main(args, resize, resize_dims):
    model = load_model(args)

    # loop all folders
    data_root = args.data_dir
    
    all_scenes_list = sorted(os.listdir(data_root))
    print(f"There are {len(all_scenes_list)} scenes in {data_root}")

    cam_name = "CAM_FRONT"
    for scene_name in tqdm(all_scenes_list):
        scene_dir = osp.join(data_root, scene_name, cam_name)
        
        args.data_dir = scene_dir
        run_exhaustive_flow(model, args, resize, resize_dims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()

    resize = True
    resize_dims = 800, 450  # W, H
    main(args, resize, resize_dims)
