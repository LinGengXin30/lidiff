import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import (
    clusterize_pcd, visualize_pcd_clusters, point_set_to_coord_feats, 
    overlap_clusters, aggregate_pcds, load_poses, apply_transform, undo_transform
)
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import point_set_to_sparse_refine
import os
import numpy as np
import MinkowskiEngine as ME

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, scan_window, seqs, split, resolution, num_points, mode):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_views'

        self.n_clusters = 50
        self.resolution = resolution
        self.scan_window = scan_window
        self.num_points = num_points
        self.seg_batch = True

        self.split = split
        self.seqs = seqs
        self.mode = mode

        # Cache for poses to avoid repeated loading
        self.pose_cache = {}
        
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            
            for file_num in range(0, len(point_seq_bin)):
                # we guarantee that the end of sequence will not generate single scans as aggregated pcds
                end_file = file_num + self.scan_window if len(point_seq_bin) - file_num > 1.5 * self.scan_window else len(point_seq_bin)
                self.points_datapath.append([os.path.join(point_seq_path, point_file) for point_file in point_seq_bin[file_num:end_file] ])
                if end_file == len(point_seq_bin):
                    break

        #self.points_datapath = self.points_datapath[:200]

    def transforms(self, points):
        points = points[None,...]

        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return points[0]

    def __getitem__(self, index):
        #index = 500
        seq_num = self.points_datapath[index][0].split('/')[-3]
        fname = self.points_datapath[index][0].split('/')[-1].split('.')[0]

        #t_frame = np.random.randint(len(self.points_datapath[index]))
        t_frame = int(len(self.points_datapath[index]) / 2)
        p_full, p_part = self.aggregate_pcds_with_cache(self.points_datapath[index], t_frame)

        p_concat = np.concatenate((p_full, p_part), axis=0)
        p_gt = p_concat.copy()
        p_concat = self.transforms(p_concat) if self.split == 'train' else p_concat

        p_full = p_concat.copy()
        p_noise = jitter_point_cloud(p_concat[None,:,:3], sigma=.2, clip=.3)[0]
        dist_noise = np.power(p_noise, 2)
        dist_noise = np.sqrt(dist_noise.sum(-1))

        _, mapping = ME.utils.sparse_quantize(coordinates=p_full / 0.1, return_index=True)
        p_full = p_full[mapping]
        dist_full = np.power(p_full, 2)
        dist_full = np.sqrt(dist_full.sum(-1))

        return point_set_to_sparse_refine(
            p_full[dist_full < 50.],
            p_noise[dist_noise < 50.],
            self.num_points*2,
            self.num_points,
            self.resolution,
            self.points_datapath[index],
        )                                       

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data
    
    def aggregate_pcds_with_cache(self, data_batch, t_frame):
        """Optimized version of aggregate_pcds with pose caching"""
        # load empty pcd point cloud to aggregate
        pcd_full = np.empty((0,3))
        pcd_part = None

        # define "namespace"
        seq_num = data_batch[0].split('/')[-3]
        fname = data_batch[0].split('/')[-1].split('.')[0]

        # load poses from cache if available
        datapath = data_batch[0].split('velodyne')[0]
        cache_key = os.path.join(datapath, 'poses.txt')
        
        if cache_key not in self.pose_cache:
            self.pose_cache[cache_key] = load_poses(
                os.path.join(datapath, 'calib.txt'), 
                os.path.join(datapath, 'poses.txt')
            )
        
        poses = self.pose_cache[cache_key]

        for t in range(len(data_batch)):
            # load the next t scan and aggregate
            fname = data_batch[t].split('/')[-1].split('.')[0]

            # load the next t scan, apply pose and aggregate
            p_set = np.fromfile(data_batch[t], dtype=np.float32)
            p_set = p_set.reshape((-1, 4))[:,:3]

            label_file = data_batch[t].replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32)
            l_set = l_set.reshape((-1))
            l_set = l_set & 0xFFFF

            # remove moving points
            static_idx = l_set < 252
            p_set = p_set[static_idx]

            # remove flying artifacts
            dist = np.power(p_set, 2)
            dist = np.sqrt(dist.sum(-1))
            p_set = p_set[dist > 3.5]

            pose_idx = int(fname)
            p_set = apply_transform(p_set, poses[pose_idx])

            if t == t_frame:
                # will be aggregated later to the full pcd
                pcd_part = p_set.copy()
            else:
                pcd_full = np.vstack([pcd_full, p_set])

        # get start position of each aggregated pcd
        pose_idx = int(fname)
        pcd_full = undo_transform(pcd_full, poses[pose_idx])
        pcd_part = undo_transform(pcd_part, poses[pose_idx])

        return pcd_full, pcd_part

##################################################################################################