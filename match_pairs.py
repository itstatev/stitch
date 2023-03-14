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
def match_pairs(pair, pairs='', resize=[360, 240], superglue='outdoor', max_keypoints=1024, keypoint_threshold=0.005, nms_radius=4, sinkhorn_iterations=20, match_threshold=0.00009, viz=False, eval=False, fast_viz=False, viz_extension=False, opencv_display=False, force_cpu=False):
    # print('the pair', pair)
    # cv2.imshow('img', pair[0])
    # cv2.waitKey(0)
    torch.set_grad_enabled(False)
    assert not (opencv_display and not viz), 'Must use --viz with --opencv_display'
    assert not (opencv_display and not fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (fast_viz and not viz), 'Must use --viz with --fast_viz'
    assert not (fast_viz and viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    print(pair[0].shape)

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
    print('after matching', pair[0].shape)

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
    # print('mta')
    # print(inp0[0], inp0[0].shape)
    inp0 = torch.moveaxis(inp0[0], 3, 1)
    inp0 = torch.moveaxis(inp0, 0, 1)
    inp1 = torch.moveaxis(inp1[0], 3, 1)
    inp1 = torch.moveaxis(inp1, 0, 1)
    # print(inp0.shape)
    # input()
    timer.update('load_image')
    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
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

    # Keep the matching keypoints.
    # valid = matches > -1
    # mkpts0 = kpts0[valid]
    # mkpts1 = kpts1[matches[valid]]
    # mconf = conf[valid]

#     if do_eval:
#         # Estimate the pose and compute the pose error.
#         assert len(pair) == 38, 'Pair does not have ground truth info'
#         K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
#         K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
#         T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

#         # Scale the intrinsics to resized image.
#         K0 = scale_intrinsics(K0, scales0)
#         K1 = scale_intrinsics(K1, scales1)

#         # Update the intrinsics + extrinsics if EXIF rotation was found.
#         if rot0 != 0 or rot1 != 0:
#             cam0_T_w = np.eye(4)
#             cam1_T_w = T_0to1
#             if rot0 != 0:
#                 K0 = rotate_intrinsics(K0, image0.shape, rot0)
#                 cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
#             if rot1 != 0:
#                 K1 = rotate_intrinsics(K1, image1.shape, rot1)
#                 cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
#             cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
#             T_0to1 = cam1_T_cam0

#         epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
#         correct = epi_errs < 5e-4
#         num_correct = np.sum(correct)
#         precision = np.mean(correct) if len(correct) > 0 else 0
#         matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

#         thresh = 1.  # In pixels relative to resized image size.
#         ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
#         if ret is None:
#             err_t, err_R = np.inf, np.inf
#         else:
#             R, t, inliers = ret
#             err_t, err_R = compute_pose_error(T_0to1, R, t)

#         # Write the evaluation results to disk.
#         out_eval = {'error_t': err_t,
#                     'error_R': err_R,
#                     'precision': precision,
#                     'matching_score': matching_score,
#                     'num_correct': num_correct,
#                     'epipolar_errors': epi_errs}
#         np.savez(str(eval_path), **out_eval)
#         timer.update('eval')

#     if do_viz_eval:
#         # Visualize the evaluation results for the image pair.
#         color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
#         color = error_colormap(1 - color)
#         deg, delta = ' deg', 'Delta '
#         if not fast_viz:
#             deg, delta = '°', '$\\Delta$'
#         e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
#         e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
#         text = [
#             'SuperGlue',
#             '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
#             'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
#         ]
#         if rot0 != 0 or rot1 != 0:
#             text.append('Rotation: {}:{}'.format(rot0, rot1))

#         # Display extra parameter info (only works with --fast_viz).
#         k_thresh = matching.superpoint.config['keypoint_threshold']
#         m_thresh = matching.superglue.config['match_threshold']
#         small_text = [
#             'Keypoint Threshold: {:.4f}'.format(k_thresh),
#             'Match Threshold: {:.2f}'.format(m_thresh),
#             'Image Pair: {}:{}'.format(stem0, stem1),
#         ]

#         make_matching_plot(
#             image0, image1, kpts0, kpts1, mkpts0,
#             mkpts1, color, text, viz_eval_path,
#             show_keypoints, fast_viz,
#             opencv_display, 'Relative Pose', small_text)

#         timer.update('viz_eval')   

#     timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

# if eval:
#     # Collate the results into a final table and print to terminal.
#     pose_errors = []
#     precisions = []
#     matching_scores = []
#     for pair in pairs:
#         name0, name1 = pair[:2]
#         stem0, stem1 = Path(name0).stem, Path(name1).stem
#         eval_path = output_dir / \
#             '{}_{}_evaluation.npz'.format(stem0, stem1)
#         results = np.load(eval_path)
#         pose_error = np.maximum(results['error_t'], results['error_R'])
#         pose_errors.append(pose_error)
#         precisions.append(results['precision'])
#         matching_scores.append(results['matching_score'])
#     thresholds = [5, 10, 20]
#     aucs = pose_auc(pose_errors, thresholds)
#     aucs = [100.*yy for yy in aucs]
#     prec = 100.*np.mean(precisions)
#     ms = 100.*np.mean(matching_scores)
#     print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
#     print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
#     print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
#         aucs[0], aucs[1], aucs[2], prec, ms))