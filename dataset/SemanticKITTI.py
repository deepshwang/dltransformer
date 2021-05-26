import torch
import numpy as np
import yaml
import os
import h5py
import pdb
from configs.config import load_config_data
from torch.utils.data import DataLoader


class SKDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_config, split_type='train', label_mapping="./config/label_mapping/semanticKitti_labelmapconfig.yaml"):
		with open(label_mapping, 'r') as stream:
			semkittiyaml = yaml.safe_load(stream)
		self.learning_map = semkittiyaml['learning_map']
		self.split_type = split_type
		if split_type == 'train':
			split = semkittiyaml['split']['train']
		elif split_type == 'val':
			split = semkittiyaml['split']['valid']
		elif split_type == 'test':
			split = semkittiyaml['split']['test']
		else:
			raise Exception('Split must be train/val/test')

		data_path = dataset_config['pc_dataset_params']['data_path']
		self.im_idx = []
		for i_folder in split:
			self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

	def __len__(self):
		'''
		Denotes the total number of samples
		'''
		return len(self.im_idx)

	def __getitem__(self, index):
		raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
		if self.split_type == 'test':
			annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
		else:
			annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
										 dtype=np.int32).reshape((-1, 1))
			annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
			annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

		data_tuple = (torch.from_numpy(raw_data[:, :3]), torch.from_numpy(annotated_data.astype(np.uint8)))
		return data_tuple

	def collate_fn(self, samples):
		points, labels = map(list, zip(*samples))
		size_list = []
		for i, point in enumerate(points):
			N, _ = point.shape
			size_list.append(N)
			perm = torch.randperm(N)
			points[i] = point[perm]
			labels[i] = labels[i][perm]
		N_min = min(size_list)
		points = [point[:N_min]for point in points]
		labels = [label[:N_min]for label in labels]

		labels = torch.stack(labels).long()
		points = torch.stack(points)

		return points, labels



	def absoluteFilePaths(self, directory):
		for dirpath, _, filenames in os.walk(directory):
			for f in filenames:
				yield os.path.abspath(os.path.join(dirpath, f))


def SKDataLoader(args):
	dataset_config = load_config_data(args.dataset_config)

	train_pcdataset = SKDataset(dataset_config=dataset_config, 
												 split_type='train',
												 label_mapping=args.labelmap_config)

	train_dataloader = DataLoader(dataset=train_pcdataset,
								  batch_size=args.batch_size,
								  shuffle=False,
								  num_workers=args.num_workers,
								  collate_fn=train_pcdataset.collate_fn)
	return train_dataloader
