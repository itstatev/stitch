from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
import cv2

from .models.matching import Matching
from .models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


# default input path: ='/home/tatev/Documents/change_detection/CD_via_segmentation/modules/stitch/pairs.txt'
def match_pairs(pair, layers_of_first_img, layers_of_second_img, pairs='', resize=[360, 240], superglue='outdoor', max_keypoints=8, keypoint_threshold=0.000000001, nms_radius=4, sinkhorn_iterations=20, match_threshold=0.00000009, viz=False, eval=False, fast_viz=False, viz_extension=False, opencv_display=False, force_cpu=False):
    # print('the pair', pair)
    # cv2.imshow('img', pair[0])
    # cv2.waitKey(0)
    torch.set_grad_enabled(False)
    assert not (opencv_display and not viz), 'Must use --viz with --opencv_display'
    assert not (opencv_display and not fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (fast_viz and not viz), 'Must use --viz with --fast_viz'
    assert not (fast_viz and viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    print(keypoint_threshold)

    if len(resize) == 2 and resize[1] == -1:
        resize = resize[0:1]
    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    print(pair[0].shape)
    # with open(input_pairs, 'r') as f:
    #     pairs = [l.split() for l in f.readlines()]
    if eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(pair))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    
    matching = Matching(config).eval().to(device)

    timer = AverageTimer(newline=True)

    keypoints0 = []
    keypoints1 = []
    matchess = []

    # Handle --cache logic.
    do_match = True
    do_eval = eval
    do_viz = viz
    do_viz_eval = eval and viz


    # Load the image pair.
    # image0 = pair[0]
    # image1 = pair[1]
    # print(image0.shape)
    # image0 = np.array(image0)
    # image1 = np.array(image1)

    # image0 = np.reshape(image0, (image0.shape[2], 1, image0.shape[0], image0.shape[1]))
    # image1 = np.reshape(image1, (image1.shape[2], 1, image1.shape[0], image1.shape[1]))
    # print('reshaped', image0.shape)
    # inp0 = torch.from_numpy(image0).float()
    # inp1 = torch.from_numpy(image1).float()

    image0, inp0, scale0 = read_image(pair[0], device, resize, 0, True)
    image1, inp1, scale1 = read_image(pair[1], device, resize, 0, True)

    inp0 = torch.moveaxis(inp0[0], 3, 1)
    inp0 = torch.moveaxis(inp0, 0, 1)
    inp1 = torch.moveaxis(inp1[0], 3, 1)
    inp1 = torch.moveaxis(inp1, 0, 1)

    timer.update('load_image')
    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1}, layers_of_first_img, layers_of_second_img)
        # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        for k, v in pred.items():
            v = v.cpu().numpy()
        print('pred', pred)
        input()
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        keypoints0.append(kpts0)
        keypoints1.append(kpts1)
        matchess.append(matches)
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': keypoints0, 'keypoints1': keypoints1,
                        'matches': matchess, 'match_confidence': conf}

    return out_matches