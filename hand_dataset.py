import numpy as np
import pickle
import utils
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import os
from albumentations import ShiftScaleRotate, Compose
import itertools


def test_sample():
    with open('/Users/pengyuehou/Downloads/datasets/Task 1/train_instances.pkl', 'rb') as f:
        instances = pickle.load(f)
    instances = instances[0:1000]
    print(instances)
    with open('/Users/pengyuehou/Downloads/datasets/Task 1/train_instances_tmp.pkl', 'wb') as f:
        pickle.dump(instances, f)


def pickle_save():
    bbs = []
    with open('/home/data/hands2019_challenge/task1/test_bbs.txt') as f:
        data = f.readlines()
        for line in data:
            data = line.split()
            for i in range(1, 5):
                data[i] = int(data[i])
            bbs.append([data[0], data[1:5]])
        print(bbs[0:5])
    with open('/home/data/hands2019_challenge/task1/test_instances.pkl','wb') as f:
        pickle.dump(bbs, f)


class Task1Dataset(Dataset):
    def __init__(self, mode, bbs_length, padding,
                 store_dir='/Users/pengyuehou/Downloads/datasets/Task 1', transform=None):
        """
        Args:
            store_dir (string): directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.padding = padding
        self.bbs_length = bbs_length
        # self.max_depth = max_depth
        self.mode = mode
        self.store_dir = store_dir
        self.camera_cfg = (475.065948, 475.065857, 315.944855, 245.287079, 640, 480)  # fx, fy, cx, cy, w, h
        self.img_size = (480, 640)
        self.K_depth = np.array([[475.065948, 0, 315.944855, 0],
                                 [0, 475.065857, 245.287079, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        self.inv_K_depth = np.linalg.inv(self.K_depth[0:3, 0:3])
        with open('%s/train_instances.pkl' % self.store_dir, 'rb') as f:
            # list of lists(filename, pose, bbs)
            self.instances = pickle.load(f)
        print('%s/%s_instances.pkl, %i instances' % (self.store_dir, self.mode, len(self.instances)))
        if self.mode == 'test':
            self.test_instances = self.instances[:(len(self.instances)//10)]
        elif self.mode == 'train':
            self.train_instances = self.instances[(len(self.instances)//10):]
        else:
            print('wrong mode')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        idx = idx
        image_name, pose, bbs = self.instances[idx]
        pose = np.reshape(pose, (-1, 3))
        pose = utils.xyz2uvd(pose, self.camera_cfg)
        # (480, 640)
        depth_img = cv2.imread('%s/train_images/%s' % (self.store_dir, image_name), cv2.IMREAD_UNCHANGED)
        # img_cropped, center, pose_cropped, v_length, u_length, ratio = self.crop_refine(depth_img, bbs, pose)
        # img_cropped = (img_cropped - min_depth) / (max_depth - min_depth)
        # img_cropped = np.expand_dims(img_cropped, axis=0)
        if self.transform is not None:
            uv = pose[:, 0:2]
            d = pose[:, 2:3]
            img, uv = self.transform(depth_img, uv)
            pose = np.concatenate([np.asarray(uv), d], axis=1)
        ratio, bbs_center, img_cropped, pose_cropped, max_depth, min_depth = self.center_ratio(depth_img,
                                                                                               idx, bbs, pose)
        # square_pose = pose_cropped[:, 0:2] - bbs_center
        # print(-100 <= square_pose.all() <= 100)
        # flag = (-100 <= square_pose <= 100).astype(np.int32)
        # print(flag)

        return img_cropped, pose_cropped

    def crop_refine(self, depth_img, bbs, pose):
        # bbs (w1, h1, w2, h2)
        w_min, h_min, w_max, h_max = bbs
        img_cropped = depth_img[h_min:h_max, w_min:w_max]
        bbs_center = np.array([(w_min + w_max) / 2, (h_min + h_max) / 2])
        bbs_center_cropped = bbs_center - [w_min, h_min]
        pose_cropped = pose - [w_min, h_min, 0]
        pose_bound = np.max(pose_cropped, axis=0)
        top = pose_bound[1]
        right = pose_bound[0]
        pose_bound = np.min(pose_cropped, axis=0)
        bottom = pose_bound[1]
        left = pose_bound[0]
        h_center = bbs_center_cropped[1]
        w_center = bbs_center_cropped[0]

        top_distance = top - h_center
        bottom_distance = h_center - bottom
        right_distance = right - w_center
        left_distance = w_center - left
        img_refined = img_cropped[int(bottom - self.padding):int(top + self.padding),
                                  int(left - self.padding):int(right + self.padding)]
        pose_refined = pose_cropped - [(left-self.padding), (bottom-self.padding), 0]
        # bound = [top_distance, bottom_distance, right_distance, left_distance]
        if top_distance >= bottom_distance:
            vertical = 2 * top_distance
        else:
            vertical = 2 * bottom_distance
        if right_distance >= left_distance:
            horizontal = 2 * right_distance
        else:
            horizontal = 2 * bottom_distance
        fillup = vertical - horizontal
        # if fillup >= 0:
        #     img_refined = np.pad(img_refined, ())
        # u:w, v:h
        ratio = np.array([[horizontal/(w_max - w_min), vertical/(h_max - h_min)]])


        # utils.plot_points_in_2d(points, None)
        return img_refined, bbs_center, pose_refined, vertical, horizontal, ratio

    def center_ratio(self, depth_img, idx, bbs, pose):
        # bbs (w1, h1, w2, h2)
        w_min, h_min, w_max, h_max = bbs
        # img_bbs = depth_img[h_min:h_max, w_min:w_max]
        # pose_cropped = pose - [w_min, h_min, 0]
        bbs_center = np.array([(w_min + w_max) // 2, (h_min + h_max) // 2])

        # pose_bound = np.max(pose, axis=0)
        # top = pose_bound[1]
        # right = pose_bound[0]
        #
        # pose_bound = np.min(pose, axis=0)
        # bottom = pose_bound[1]
        # left = pose_bound[0]

        max_pose = np.amax(pose, axis=0)
        min_pose = np.amin(pose, axis=0)

        pose_center = np.mean(pose, axis=0)[0:2]
        # # check if hand exceed bbs
        # if top <= h_max and right <= w_max and left >= w_min and bottom >= h_min:
        #     pass
        # else:
        #     print('%s out of bbs' % idx)

        w_min = int(bbs_center[0] - self.bbs_length)
        w_max = int(bbs_center[0] + self.bbs_length)
        h_min = int(bbs_center[1] - self.bbs_length)
        h_max = int(bbs_center[1] + self.bbs_length)

        # shift if out of img
        if w_min <= 0:
            w_max = w_max - w_min
            w_min = 0
            pose[0] = pose[0] - w_min
        if w_max >= self.img_size[1]:
            w_min = w_min - (w_max - self.img_size[1])
            w_max = self.img_size[1]
            pose[0] = pose[0] - (w_max - self.img_size[1])
        if h_min <= 0:
            h_max = h_max - h_min
            h_min = 0
            pose[1] = pose[1] - h_min
        if h_max >= self.img_size[0]:
            h_min = h_min - (h_max - self.img_size[0])
            h_max = self.img_size[0]
            pose[1] = pose[1] - (h_max - self.img_size[0])

        # img_cropped = depth_img[h_min:h_max, w_min:w_max]

        pose_cropped = pose - [w_min, h_min, 0]
        img_cropped = depth_img[h_min:h_max, w_min:w_max]

        ratio = np.array([(bbs_center[0] - pose_center[0])/(2 * self.bbs_length),
                         (bbs_center[1] - pose_center[1])/(2 * self.bbs_length)])
        max_depth, min_depth = np.max(depth_img[depth_img > 0]), np.min(depth_img[depth_img > 0])

        return ratio, bbs_center, img_cropped, pose_cropped, max_depth, min_depth


class Augmentation(object):
    def __init__(self, shift_limit=0.01, scale_limit=0.2, rotate_limit=180, apply_prob=1.0):
        self.aug = Compose([ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit,
                                             rotate_limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, value=0)],
                           p=apply_prob, keypoint_params={'format': 'xy'})

    def __call__(self, img, keypoints):
        data = {'image': img, 'keypoints': keypoints}
        augmented = self.aug(**data)
        return augmented['image'], augmented['keypoints']


def in_test():
    os.environ["OMP_NUM_THREADS"] = "1"
    aug = Augmentation()
    dataset = Task1Dataset(mode='train', padding=20, bbs_length=150, transform=None)
    pose = []
    ind = 0
    for i in range(10000):
        img_cropped, pose_cropped = dataset[i]
        if (0 <= pose_cropped[:, 0:2]).all() and (pose_cropped[:, 0:2] <= 300).all():
            ind+=1
        else:
            print(pose_cropped)

    print(ind)
    #     plt.figure()
    #     plt.imshow(img_cropped, cmap='gray')
    #     plt.scatter(pose_cropped[:, 0], pose_cropped[:, 1], c='r')
    #     plt.show()
    #     ratios = np.append(ratios, ratio, axis=0)
    # # print(ratios, np.max(ratios))


if __name__ == '__main__':
    # test_sample()
    in_test()
