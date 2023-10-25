'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-22 10:21:23
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import sys

sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


import numpy as np
import os
import os.path as osp
import cv2

from utils import flow_viz

DEVICE = 'cuda'


def viz(img, flo, flowpath, rfpath):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=1)

    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    cv2.imwrite(flowpath, flo)
    cv2.imwrite(rfpath, img_flo[:, :, [2,1,0]])


def load_image(imfile, resize=False, resize_dims=None):
    img_input = Image.open(imfile)
    if resize:
        assert resize_dims is not None
        img_input = img_input.resize(resize_dims)

    img = np.array(img_input).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


if __name__ == '__main__':
    data_dir = "data/swing"
    data_dir = "data/nuscenes_scene_sequence/0ac05652a4c44374998be876ba5cd6fd/CAM_FRONT"
    img_dir = osp.join(data_dir, "color")
    flow_dir = osp.join(data_dir, "raft_flow")

    img_names = sorted(os.listdir(img_dir))

    id1 = 0
    id2 = 1
    img_name1 = img_names[id1]
    img_name2 = img_names[id2]

    resize_dims = [800, 450]
    img1 = load_image(osp.join(img_dir, img_name1), resize=True, resize_dims=resize_dims)
    img2 = load_image(osp.join(img_dir, img_name2), resize=True, resize_dims=resize_dims)

    flow_file = os.path.join(flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
    flow = np.load(flow_file)
    print(img1.shape, img2.shape, flow.shape)

    flow_up = torch.from_numpy(flow).permute(2, 0, 1).to(DEVICE)
    flow_up = flow_up[None]

    save_dir = "./results"
    save_flow_path = osp.join(save_dir, f"flow_{id1}_{id2}.png")
    save_raw_flow_path = osp.join(save_dir, f"raw_flow_{id1}_{id2}.png")

    viz(img1, flow_up, save_flow_path, save_raw_flow_path)