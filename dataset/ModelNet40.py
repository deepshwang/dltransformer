import os
import os.path as osp
import shlex
import shutil
import subprocess

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
import tqdm
import pdb
# import utils.data_utils as d_utils
import msgpack


DATA_DIR = '/media/TrainDataset'


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40Dataset(data.Dataset):
    def __init__(self, num_points=1024, transforms=None, train=True, download=True):
        super().__init__()
        self.ds_ratio=4
        self.cluster_filelist, self.downsample_filelist = self._get_index_file_list(train)

        self.transforms = transforms

        self.set_num_points(num_points)
        self._cache = os.path.join(DATA_DIR, "modelnet40_normal_resampled_cache")

        if not osp.exists(self._cache):
            self.folder = "modelnet40_normal_resampled"
            self.data_dir = os.path.join(DATA_DIR, self.folder)
            self.url = (
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
            )

            if download and not os.path.exists(self.data_dir):
                zipfile = os.path.join(DATA_DIR, os.path.basename(self.url))
                subprocess.check_call(
                    shlex.split("curl {} -k -o {}".format(self.url, zipfile))
                )

                subprocess.check_call(
                    shlex.split("unzip {} -d {}".format(zipfile, DATA_DIR))
                )

                subprocess.check_call(shlex.split("rm {}".format(zipfile)))

            self.train = train
            self.set_num_points(num_points)

            self.catfile = os.path.join(self.data_dir, "modelnet40_shape_names.txt")
            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            os.makedirs(self._cache)

            print("Converted to LMDB for faster dataloading while training")
            for split in ["train", "test"]:
                if split == "train":
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet40_train.txt")
                        )
                    ]
                else:
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet40_test.txt")
                        )
                    ]

                shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]
                # list of (shape_name, shape_txt_file_path) tuple
                self.datapath = [
                    (
                        shape_names[i],
                        os.path.join(self.data_dir, shape_names[i], shape_ids[i])
                        + ".txt",
                    )
                    for i in range(len(shape_ids))
                ]

                with lmdb.open(
                    osp.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    for i in tqdm.trange(len(self.datapath)):
                        fn = self.datapath[i]
                        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

                        pt_idxs = np.arange(0, self.num_points)
                        np.random.shuffle(pt_idxs)

                        point_set = point_set[pt_idxs, :]

                        cls = self.classes[self.datapath[i][0]]
                        cls = int(cls)

                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(
                                dict(pc=point_set, lbl=cls), use_bin_type=True
                            ),
                        )

            shutil.rmtree(self.data_dir)

        self._lmdb_file = osp.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)

        point_set = ele["pc"].copy()

        ########## Because a fixed random set has already been stored 
        ########## (For constant k-means clustering index) 

        # pt_idxs = np.arange(0, self.num_points)
        # np.random.shuffle(pt_idxs)

        # point_set = point_set[pt_idxs, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.transforms is not None:
            point_set = self.transforms(point_set)

        with open(self.cluster_filelist[idx], "rb") as c_file:
            c_data = c_file.read()
        cluster_idx = msgpack.unpackb(c_data, strict_map_key=False)

        with open(self.downsample_filelist[idx], "rb") as d_file:
            d_data = d_file.read()
        downsample_idx = msgpack.unpackb(d_data, strict_map_key=False, object_hook=msgpack_numpy.decode)
        ds_pre = torch.from_numpy(downsample_idx.copy())


        return point_set, ele["lbl"], cluster_idx, ds_pre

    def __len__(self):
        return self._len


    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


    def _get_index_file_list(self, train=True):
        index_root = '/media/TrainDataset/modelnet40_normal_resampled_cache/index_files'
        if train:
            index_root = index_root + "/train"
        else:
            index_root = index_root + '/test'
        cluster_filelist = []
        downsample_filelist =[]
        for f in os.listdir(index_root):
            # num = f.split(".")[0].split("_")[-1]
            # if len(num) < 4:
            #   z_pad = '0' * (4-len(num))
            #   num_new = z_pad + num
            #   new_f = f.split("_")[0] + "_" + num_new + ".msgpack"
            #   os.rename(os.path.join(index_root, f), os.path.join(index_root, new_f))

            if f[-7:] == 'msgpack':
                if f[:4] == 'clus':
                    cluster_filelist.append(os.path.join(index_root, f))
                elif f[:4] == 'down':
                    downsample_filelist.append(os.path.join(index_root, f))
        cluster_filelist.sort()
        downsample_filelist.sort()
        return cluster_filelist, downsample_filelist

    def collate_fn(self, samples):
        point_set, labels, cluster_idxs, downsample_idxs = map(list, zip(*samples))
        # print("point_set: ")
        # print(point_set)
        # print("point_set type: ")
        # print(type(point_set))
        # print("point_set length: ")
        # print(len(point_set))
        # print("point_set component")
        # print(point_set[0])
        point_set = torch.stack(point_set)
        labels = torch.Tensor(labels).long()

        cluster_idx_list = []
        for i in range(len(cluster_idxs[0])):
            cluster_idx_list.append([batch[i] for batch in cluster_idxs])

        ds_pre_1 = torch.stack(downsample_idxs)
        ds_pre_2 = torch.arange(ds_pre_1.shape[2]/self.ds_ratio).expand(ds_pre_1.shape[0], 1, -1).type(torch.IntTensor)
        ds_pre_3 = torch.arange(ds_pre_2.shape[2]/self.ds_ratio).expand(ds_pre_2.shape[0], 1, -1).type(torch.IntTensor)
        ds_pre_4 = torch.arange(ds_pre_3.shape[2]/self.ds_ratio).expand(ds_pre_3.shape[0], 1, -1).type(torch.IntTensor)


        return point_set, labels, cluster_idx_list, [ds_pre_1, ds_pre_2, ds_pre_3, ds_pre_4]




def ModelNet40DataLoader(args, num_points, shuffle, train, transforms):
    dataset = ModelNet40Dataset(num_points, train=train, transforms=transforms)
    if train:
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size
    dataloader = data.DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    shuffle=shuffle,
                    collate_fn=dataset.collate_fn)
    return dataloader


if __name__ == "__main__":
    from torchvision import transforms

    # transforms = transforms.Compose(
    #     [
    #         d_utils.PointcloudToTensor(),
    #         d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
    #         d_utils.PointcloudScale(),
    #         d_utils.PointcloudTranslate(),
    #         d_utils.PointcloudJitter(),
    #     ]
    # )
    dset = ModelNet40Dataset(1024, train=True, transforms=None)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)