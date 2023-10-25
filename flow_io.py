'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-10-22 09:45:37
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import numpy as np


def write_flow(flow: np.ndarray, flow_file: str) -> None:
    """Write the flow in disk.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        flow (ndarray): The optical flow that will be saved.
        flow_file (str): The file for saving optical flow.
    """

    with open(flow_file, 'wb') as f:
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)



if __name__ == "__main__":
    input_file = "data/nuscenes_scene_sequence/0ac05652a4c44374998be876ba5cd6fd/CAM_FRONT/raft_flow/n008-2018-09-18-14-35-12-0400__CAM_FRONT__1537295813862404.jpg_n008-2018-09-18-14-35-12-0400__CAM_FRONT__1537295814362404.jpg.npy"
    data = np.load(input_file)
    print(data.shape)

    np.savez_compressed("test.npz", data)
