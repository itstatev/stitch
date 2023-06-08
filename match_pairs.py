import torch


from .models.matching import Matching
from .models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


def match_pairs(pair, layers_of_first_img, layers_of_second_img, pairs='', resize=[-1], superglue='outdoor', max_keypoints=8, keypoint_threshold=0.000000001, nms_radius=4, sinkhorn_iterations=20, match_threshold=0.00000009, viz=False, eval=False, fast_viz=False, viz_extension=False, opencv_display=False, force_cpu=False):
    torch.set_grad_enabled(False)
    assert not (opencv_display and not viz), 'Must use --viz with --opencv_display'
    assert not (opencv_display and not fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (fast_viz and not viz), 'Must use --viz with --fast_viz'
    assert not (fast_viz and viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

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


    _, inp0, _ = read_image(pair[0], device, resize, 0, True)
    _, inp1, _ = read_image(pair[1], device, resize, 0, True)

    inp0 = torch.moveaxis(inp0[0], 3, 1)
    inp0 = torch.moveaxis(inp0, 0, 1)
    inp1 = torch.moveaxis(inp1[0], 3, 1)
    inp1 = torch.moveaxis(inp1, 0, 1)

    timer.update('load_image')
    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1}, layers_of_first_img, layers_of_second_img)
        pred = {k: v.cpu().numpy() for k, v in pred.items()}
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