import os

import numpy as np
import torch as th

from torch.utils import data
import tools



def trace(cov):
    return cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

class DatasetNTU60SPD(data.Dataset):
    def __init__(self, path, names, p_interval=1, random_rot=False, window_size=-1, bone=False, vel=False):
        """
                :param path:
                :param names:
                :param random_rot: rotate skeleton around xyz axis
                :param window_size: The length of the output sequence
                :param bone: use bone modality or not
                :param vel: use motion modality or not
        """

        self._path = path
        self._names = names

        self.window_size = window_size
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        data_numpy = th.load( os.path.join(self._path, self._names[item]))
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if not th.is_tensor(data_numpy):
            data_tensor = th.from_numpy(data_numpy)
        else:
            # If it's already a tensor, use it directly
            data_tensor = data_numpy

        y = int(self._names[item].split('.')[0].split('_')[-1])
        y = th.from_numpy(np.array(y)).long()
        X_new = self.select_main_skeleton(data_tensor)
        return X_new, y

    def select_main_skeleton(self,skeleton_sample):
        """
        Selects the main skeleton based on the highest sum of variances of the X, Y, and Z values
        across all joints for each person (skeleton). https://github.com/shahroudy/NTURGB-D
        Parameters:
        - skeleton_sample: A tensor of shape [c, t, j, s], where
            c is the number of coordinates (3 for X, Y, Z),
            t is the number of frames,
            j is the number of joints,
            s is the number of people (skeletons).
        Returns:
        - main_skeleton: The main skeleton as a tensor of shape [c, t, j].
    """
        # Compute the variance across frames for each coordinate of each joint of each skeleton
        variances = skeleton_sample.var(dim=1)  # Variance across frames

        # Sum the variances of all coordinates (X, Y, Z) for each joint, across all skeletons, in one operation
        total_variances = variances.sum(dim=(0, 1))  # Sum across coordinates 'c' and joints 'j'

        # Identify the index of the skeleton with the highest total variance
        main_skeleton_index = total_variances.argmax()

        # Select the main skeleton
        main_skeleton = skeleton_sample[..., main_skeleton_index]

        return main_skeleton


train_feeder_args = {
    # 'data_path': '/data/nturgbd_cross_view/nturgbd_cross_view',
    # 'split': 'train',
    # 'debug': False,
    # 'random_choose': False,
    # 'random_shift': False,
    # 'random_move': False,
    'window_size': 64,
    # 'normalization': False,
    'random_rot': True,
    'p_interval': [0.5, 1],
    'vel': False,
    'bone': False
}

test_feeder_args = {
    # 'data_path': '/data/nturgbd_cross_view/nturgbd_cross_view',
    # 'split': 'test',
    'window_size': 64,
    'p_interval': [0.95],
    'vel': False,
    'bone': False,
    # 'debug': False
}


class DataLoaderNTU60:
    def __init__(self, data_path='./data/nturgbd_cross_view/nturgbd_cross_view', batch_size=256, train_feeder_args=train_feeder_args, test_feeder_args=test_feeder_args):
        path_train = os.path.join(data_path,'train')
        path_test = os.path.join(data_path,'test')
        names_train = {}
        names_test = {}
        for filenames in os.walk(path_train):
            names_train = sorted(filenames[2])
        for filenames in os.walk(path_test):
            names_test = sorted(filenames[2])
        self._train_generator = data.DataLoader(dataset=DatasetNTU60SPD(path_train, names_train,**train_feeder_args),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                drop_last=True)
        self._test_generator = data.DataLoader(DatasetNTU60SPD(path_test, names_test,**test_feeder_args),
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False)





