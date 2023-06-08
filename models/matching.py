import torch
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
        pred['keypoints0'] = []
        pred['scores0'] = []
        pred['descriptors0'] = []
        pred['keypoints1'] = []
        pred['scores1'] = []
        pred['descriptors1'] =  []

        for a_layer in layers_of_first_img:
            width, height = a_layer.shape
            mask = np.where(a_layer == 255)
            coordinates = list(zip(mask[0], mask[1]))
            top_left = min(coordinates, key=lambda c: abs(c[0]) + abs(c[1]))
            bottom_right = max(coordinates, key=lambda c: abs(c[0]) + abs(c[1]))

            top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
            bottom_right = (min(bottom_right[0] + 10, width - 1), min(bottom_right[1] + 10, height - 1))

            cropped = data['image0'].permute(0, 3, 2, 1).squeeze()[:, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cropped = (cropped.permute(2, 1, 0)*255).cpu().numpy().astype(np.uint8)

            cropped = (cropped.transpose(2, 0, 1) / 255)
            cropped = torch.from_numpy(cropped).float().unsqueeze(1).cuda()
            
            pred0 = self.superpoint({'image': cropped})

            for tensor in pred0['keypoints']:
                for keypoint in tensor:
                    keypoint = keypoint.tolist()
                    keypoint[0] = keypoint[0] + top_left[0]
                    keypoint[1] = keypoint[1] + top_left[1]
                    pred['keypoints0'].append(keypoint)

            for el in pred0['scores']:
                el = el.tolist()
                pred['scores0'].extend(el)

            for descr in pred0['descriptors']:
                if descr.shape != torch.Size([256, 1]) and descr.shape != torch.Size([256, 0]):
                    new_descr = descr.T.tolist()
                    pred['descriptors0'].extend(new_descr) 

        pred['keypoints0'] = torch.FloatTensor(pred['keypoints0']).cuda()
        pred['scores0'] = torch.FloatTensor(pred['scores0']).cuda()
        pred['descriptors0'] = torch.FloatTensor(pred['descriptors0']).T.cuda()

        for b_layer in layers_of_second_img:
            width, height = b_layer.shape
            mask = np.where(b_layer == 255)
            coordinates = list(zip(mask[0], mask[1]))
            top_left = min(coordinates, key=lambda c: abs(c[0]) + abs(c[1]))
            bottom_right = max(coordinates, key=lambda c: abs(c[0]) + abs(c[1]))

            top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
            bottom_right = (min(bottom_right[0] + 10, width - 1), min(bottom_right[1] + 10, height - 1))

            cropped = data['image1'].permute(0, 3, 2, 1).squeeze()[:, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cropped = (cropped.permute(2, 1, 0)*255).cpu().numpy().astype(np.uint8)
            cropped = (cropped.transpose(2, 0, 1) / 255)
            cropped = torch.from_numpy(cropped).float().unsqueeze(1).cuda()
            
            pred1 = self.superpoint({'image': cropped})

            for tensor in pred1['keypoints']:
                for keypoint in tensor:
                    keypoint = keypoint.tolist()
                    keypoint[0] = keypoint[0] + top_left[0]
                    keypoint[1] = keypoint[1] + top_left[1]
                    pred['keypoints1'].append(keypoint)

            for el in pred1['scores']:
                el = el.tolist()
                pred['scores1'].extend(el)

            for descr in pred1['descriptors']:
                if descr.shape != torch.Size([256, 1]) and descr.shape != torch.Size([256, 0]):
                    new_descr = descr.T.tolist()
                    pred['descriptors1'].extend(new_descr)

        pred['keypoints1'] = torch.FloatTensor(pred['keypoints1']).cuda()
        pred['scores1'] = torch.FloatTensor(pred['scores1']).cuda()
        pred['descriptors1'] = torch.FloatTensor(pred['descriptors1']).T.cuda()

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.tensor(data[k]) if 'scores' in k else torch.stack(data[k], dim=0)

        # Perform the matching
        pred = {**pred, **self.superglue(data)}
        return pred 