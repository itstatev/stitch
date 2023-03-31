# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import cv2
import numpy as np
from .superpoint import SuperPoint
from .superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data, layers_of_first_img, layers_of_second_img):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}
        pred['keypoints0'] = torch.empty((len(layers_of_first_img), 1), device='cuda')
        pred['scores0'] = torch.empty((1, len(layers_of_first_img)), device='cuda')
        pred['descriptors0'] = torch.empty((len(layers_of_first_img), 1), device='cuda')
        pred['keypoints1'] = torch.empty((len(layers_of_second_img), 1), device='cuda')
        pred['scores1'] = torch.empty((1, len(layers_of_second_img)), device='cuda')
        pred['descriptors1'] = torch.empty((len(layers_of_second_img), 1), device='cuda')

        for a_layer in layers_of_first_img:
            a_layer = np.array(a_layer, np.uint8)

            # print('Along y-axis')
            x, y = np.where(a_layer == 255)
            sort = np.argsort(y)
            y_min_x, y_min_y = x[sort[0]], y[sort[0]]
            y_max_x, y_max_y = x[sort[-1]], y[sort[-1]]
            # print("Min (x, y): ({}, {})".format(y_min_x, y_min_y))
            # print("Max (x, y): ({}, {})".format(y_max_x, y_max_y))

            # print('Along x-axis')
            sort = np.argsort(x)
            x_min_x, x_min_y = x[sort[0]], y[sort[0]]
            x_max_x, x_max_y = x[sort[-1]], y[sort[-1]]
            # print("Min (x, y): ({}, {})".format(x_min_x, x_min_y))
            # print("Max (x, y): ({}, {})".format(x_max_x, x_max_y))

            min_width = (max([x_min_x - 10, 0]), x_min_y)
            max_width = (min([x_max_x  + 10, 240]), x_max_y)
            min_height = (min([y_min_x + 10, 360]), y_min_y)
            max_height = (max([y_max_x - 10, 0]), y_max_y)

            # print('min width', min_width)
            # print('max width', max_width)
            # print('min height', min_height)
            # print('max height', max_height)
            cropped = data['image0'].permute(0, 3, 2, 1).squeeze()[:, min_height[1]:max_height[1], min_width[0]:max_width[0]]
            cropped = cropped.clone().unsqueeze(1).permute(0, 1, 3, 2)
            print('Cropped shape: ', cropped.shape)

            # if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': cropped})
            # print('pred0', pred0['descriptors'][0])

            for tensor in pred0['keypoints']:
                for keypoint in tensor:
                    keypoint[0] = keypoint[0] + x_min_x
                    keypoint[1] = keypoint[1] + y_max_y
                pred['keypoints0'] = torch.cat((pred['keypoints0'], tensor), dim=0)
            
            print('keyp', pred['keypoints0'])
            input()

            for el in pred0['scores']:
                for sc in el:
                    pred['scores0'].append(sc)

            for descr in pred0['descriptors']:
                if descr.shape != torch.Size([1, 2]):
                    if descr.shape != torch.Size([256, 1]):
                        descr = torch.split(descr, 1, dim=1)
                        for i, tens in enumerate(descr):
                            if tensor.shape != torch.Size([256, 0]):
                                pred['descriptors0'].append(tens)
                    else:
                        pred['descriptors0'].append(tens) 

        # print('descriptors0', pred['descriptors0'])
        input()

        for b_layer in layers_of_second_img:
            b_layer = np.array(b_layer, np.uint8)

            # Along x-axis
            x, y = np.where(b_layer == 255)
            sort = np.argsort(y)
            y_min_x, y_min_y = x[sort[0]], y[sort[0]]
            y_max_x, y_max_y = x[sort[-1]], y[sort[-1]]

            # Along y-axis
            sort = np.argsort(x)
            x_min_x, x_min_y = x[sort[0]], y[sort[0]]
            x_max_x, x_max_y = x[sort[-1]], y[sort[-1]]

            min_width = (max([x_min_x - 10, 0]), x_min_y)
            max_width = (min([x_max_x  + 10, 240]), x_max_y)
            min_height = (min([y_min_x + 10, 360]), y_min_y)
            max_height = (max([y_max_x - 10, 0]), y_max_y)

            cropped = data['image1'].permute(0, 3, 2, 1).squeeze()[:, min_height[1]:max_height[1], min_width[0]:max_width[0]]
            cropped = (cropped.permute(2, 1, 0)*255).cpu().numpy().astype(np.uint8)
            print('cropped shape and type', cropped.shape, type(cropped))
            cv2.imwrite('cropped.jpg', cropped)

            cropped = ( cropped.transpose(2, 0, 1) / 255)
            cropped = torch.from_numpy(cropped).float().unsqueeze(1).cuda()
            print('cropped shape and type', cropped.shape, type(cropped))

            pred1 = self.superpoint({'image': cropped})
            
            for tensor in pred1['keypoints']:
                for keypoint in tensor:
                    keypoint[0] = keypoint[0] + x_min_x
                    keypoint[1] = keypoint[1] + y_max_y
                    pred['keypoints1'].append(keypoint)
            
            for el in pred1['scores']:
                for sc in el:
                    pred['scores1'].append(sc)

            for descr in pred1['descriptors']:
                if descr.shape != torch.Size([256, 1]) and descr.shape != torch.Size([256, 0]):
                    pred['descriptors1'].append(descr) 

            # print('keyp', torch.tensor(pred['keypoints1']))
            # input()
                    # descr = torch.split(descr, 1, dim=1)
                #     for i, tens in enumerate(descr):
                #         if tens.shape != torch.Size([256, 0]):
                #             pred['descriptors1'].append(tens)
                # else:
                #     pred['descriptors1'].append(tens) 
                

        print('pred keypoints1', len(pred['keypoints1']))
        print('pred scores1', len(pred['scores1']))
        print('pred descriptors1', len(pred['descriptors1']))


        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}
        # print('data', data.keys())
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k], dim=0)
                 
        # data['scores0'] = torch.unsqueeze(data['scores0'], dim=1)
        # data['scores0'] = torch.unsqueeze(data['scores0'], dim=1)
        # data['scores0'] = torch.unsqueeze(data['scores0'], dim=1)
        # data['scores1'] = torch.unsqueeze(data['scores1'], dim=1)
        # data['scores1'] = torch.unsqueeze(data['scores1'], dim=1)
        # data['scores1'] = torch.unsqueeze(data['scores1'], dim=1)
        # data['keypoints0'] = torch.unsqueeze(data['keypoints0'], dim=1)
        # data['keypoints1'] = torch.unsqueeze(data['keypoints1'], dim=1)
        # data['keypoints1'] = torch.unsqueeze(data['keypoints1'], dim=1)
        # data['keypoints1'] = torch.unsqueeze(data['keypoints1'], dim=1)
        # data['descriptors0'] = torch.unsqueeze(data['descriptors0'], dim=1)
        # data['descriptors1'] = torch.unsqueeze(data['descriptors1'], dim=1)
        
        print('data', data['scores0'].shape, data['keypoints0'].shape, data['descriptors0'].shape, data['image0'].shape)

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred 