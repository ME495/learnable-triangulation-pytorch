import os
import os.path as osp
from collections import defaultdict
import pickle

import numpy as np
import scipy.io as sio
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils import volumetric
from mvn.datasets import utils as dataset_utils
from mvn.utils.multiview import triangulate_batch_of_points, euclidean_to_homogeneous
import matplotlib.pyplot as plt


class STBMultiViewDataset(Dataset):
    """
        STB for multiview tasks.
    """

    def __init__(self,
                 stb_root='/media/sda1/dataset/stereo_hand_pose_dataset',
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="stb",
                 crop=True
                 ):
        assert train or test, '`STBMultiViewDataset` must be constructed with at least ' \
                              'one of `test=True` / `train=True`'

        self.stb_root = stb_root
        self.pred_results_path = pred_results_path
        self.image_shape = image_shape
        self.train = train
        self.test = test
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.kind = kind
        self.crop = crop

        self.fx = 822.79041
        self.fy = 822.79041
        self.u0 = 318.47345
        self.v0 = 250.31296
        self.B = 120.054
        self.K = np.array([[self.fx,      0., self.u0],
                           [     0., self.fy, self.v0],
                           [     0.,      0.,      1.]], dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.t_l = np.zeros((3, 1), dtype=np.float32)
        self.t_r = np.zeros((3, 1), dtype=np.float32)
        self.t_r[0, 0] = -self.B

        slice = [2, 3, 4, 5, 6] if self.train else [1]
        self.joint_3d, self.uv_l, self.uv_r, self.img_path_l, self.img_path_r = self.load_labels(slice)
        self.bbox_l = self.calc_bbox(self.uv_l)
        self.bbox_r = self.calc_bbox(self.uv_r)

        self.keypoints_3d_pred = None
        if pred_results_path is not None:
            pred_results = np.load(pred_results_path, allow_pickle=True)
            keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
            self.keypoints_3d_pred = keypoints_3d_pred
            assert len(self.keypoints_3d_pred) == len(self), \
                f"[train={train}, test={test}] labels has {len(self)} samples, but pred_results " + \
                f"has {len(self.keypoints_3d_pred)}. Did you follow all preprocessing instructions carefully?"

    def load_labels(self, slice):
        joint_3d_list, uv_l_list, uv_r_list, img_path_l, img_path_r = [], [], [], [], []
        labels_path = osp.join(self.stb_root, 'labels')
        img_path = osp.join(self.stb_root, 'images')
        for s in ['Counting', 'Random']:
            for i in slice:
                labels = sio.loadmat(osp.join(labels_path, f"B{i}{s}_BB.mat"))
                joint_3d = np.transpose(labels['handPara'], (2, 1, 0))
                u_l = joint_3d[..., 0] / joint_3d[..., 2] * self.fx + self.u0
                u_r = (joint_3d[..., 0]-self.B) / joint_3d[..., 2] * self.fx + self.u0
                v = joint_3d[..., 1] / joint_3d[..., 2] * self.fy + self.v0
                uv_l = np.stack([u_l, v], axis=-1)
                uv_r = np.stack([u_r, v], axis=-1)
                joint_3d_list.append(joint_3d)
                uv_l_list.append(uv_l)
                uv_r_list.append(uv_r)
                for img_id in range(joint_3d.shape[0]):
                    img_path_l.append(osp.join(img_path, f"B{i}{s}", f"BB_left_{img_id}.png"))
                    img_path_r.append(osp.join(img_path, f"B{i}{s}", f"BB_right_{img_id}.png"))
        joint_3d = np.concatenate(joint_3d_list, axis=0)
        uv_l = np.concatenate(uv_l_list, axis=0)
        uv_r = np.concatenate(uv_r_list, axis=0)
        return joint_3d, uv_l, uv_r, img_path_l, img_path_r

    def calc_bbox(self, uv):
        left = np.min(uv[..., 0], axis=-1)
        upper = np.min(uv[..., 1], axis=-1)
        right = np.max(uv[..., 0], axis=-1)
        lower = np.max(uv[..., 1], axis=-1)
        bbox = np.stack([left, upper, right, lower], axis=-1)
        return bbox

    def build_sample(self, sample, image_path, bbox, R, t, K):
        image = cv2.imread(image_path)
        retval_camera = Camera(R, t, K)

        bbox = get_square_bbox(bbox)
        bbox = scale_bbox(bbox, self.scale_bbox)

        if self.crop:
            # crop image
            image = crop_image(image, bbox)
            retval_camera.update_after_crop(bbox)

        if self.image_shape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, self.image_shape)
            retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)
            sample['image_shapes_before_resize'].append(image_shape_before_resize)

        if self.norm_image:
            image = normalize_image(image)

        sample['images'].append(image)
        sample['detections'].append(bbox + (1.0,))
        sample['cameras'].append(retval_camera)
        sample['proj_matrices'].append(retval_camera.projection)

    def __getitem__(self, item):
        sample = defaultdict(list)  # return value
        self.build_sample(sample, self.img_path_l[item], self.bbox_l[item], self.R, self.t_l, self.K)
        self.build_sample(sample, self.img_path_r[item], self.bbox_r[item], self.R, self.t_r, self.K)

        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(self.joint_3d[item],
            ((0, 0), (0, 1)), 'constant', constant_values=1.0)

        sample['indexes'] = item

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[item]

        sample.default_factory = None

        # self.show(sample['images'][0], sample['keypoints_3d'], sample['proj_matrices'][0])
        # self.show(sample['images'][1], sample['keypoints_3d'], sample['proj_matrices'][1])
        return sample

    def show(self, img, joint_3d, proj_mat):
        uv = proj_mat @ np.transpose(joint_3d)
        uv = np.transpose(uv)
        uv[:, :2] /= uv[:, 2:]
        plt.clf()
        plt.imshow(img)
        plt.scatter(uv[:, 0], uv[:, 1], c='red')
        plt.show()

    def __len__(self):
        return self.joint_3d.shape[0]

    def evaluate_using_per_pose_error(self, per_pose_error):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for k, v in action_scores.items():
                action_scores[k] = float('nan') if v['frame_count'] == 0 else (v['total_loss'] / v['frame_count'])

            return action_scores

        subject_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }


        return subject_scores

    def evaluate(self, keypoints_3d_predicted):
        keypoints_gt = self.joint_3d
        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(
                '`keypoints_3d_predicted` shape should be %s, got %s' % \
                (keypoints_gt.shape, keypoints_3d_predicted.shape))

        # mean error per 16/17 joints in mm, for each pose
        per_pose_error = np.sqrt(((keypoints_gt - keypoints_3d_predicted) ** 2).sum(2)).mean(1)

        # relative mean error per 16/17 joints in mm, for each pose
        root_index = 0

        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
        keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)

        result = {
            'per_pose_error': self.evaluate_using_per_pose_error(per_pose_error),
            'per_pose_error_relative': self.evaluate_using_per_pose_error(per_pose_error_relative)
        }

        return result['per_pose_error_relative']['Average']['Average'], result


if __name__ == '__main__':
    from tqdm import tqdm
    train_dataset = STBMultiViewDataset(train=True, scale_bbox=1.2,  image_shape=(384, 384))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=False,
                                                 min_n_views=None,
                                                 max_n_views=None),
        num_workers=4,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )
    k3d = []
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = \
            dataset_utils.prepare_batch(batch_data, torch.device("cpu"), None)
        # print(images_batch.shape)
        # print(keypoints_3d_gt.shape)
        # print(keypoints_3d_validity_gt.shape)
        # print(proj_matricies_batch.shape)
        # k3d_h = torch.cat([keypoints_3d_gt, torch.ones(keypoints_3d_gt.shape[:2]+(1,), dtype=torch.float32)], dim=-1)
        # k3d_h = torch.transpose(k3d_h, -2, -1)
        # uv_h = proj_matricies_batch @ k3d_h[:, None, :, :]
        # uv_h = torch.transpose(uv_h, -2, -1)
        # uv_h = uv_h / uv_h[..., -1:]
        # uv = uv_h[..., :-1]
        # keypoints_3d_pred = triangulate_batch_of_points(proj_matricies_batch, uv)
        # print(keypoints_3d_pred[0, 0])
        # print(keypoints_3d_gt[0, 0])
        # print(uv[0, 0])
        k3d.append(keypoints_3d_gt)
    k3d = torch.cat(k3d, dim=0).reshape((-1, 3))
    print(k3d.max(0))
    print(k3d.min(0))
    # test_dataset = STBMultiViewDataset(test=True)
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     collate_fn=dataset_utils.make_collate_fn(randomize_n_views=False,
    #                                              min_n_views=None,
    #                                              max_n_views=None),
    #     num_workers=4,
    #     worker_init_fn=dataset_utils.worker_init_fn,
    #     pin_memory=True
    # )
    # for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
    #     images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = \
    #         dataset_utils.prepare_batch(batch_data, torch.device("cuda:0"), None)
