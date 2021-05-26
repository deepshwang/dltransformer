from dataset.SemanticKITTI import SKDataset
from dataset.S3DIS import S3DISDataLoader
from models.layers import DLPTLayer, DLPTBlock
from models.modules import DLPTNet_cls
from utils.argparse import argument_parser
from utils.util import open_yaml
from utils.visualize import draw_points_with_labels, draw_points_without_labels
from dataset.ModelNet40 import ModelNet40Dataset, ModelNet40DataLoader
import utils.data_utils as d_utils



import torch
import pickle
import pdb
import gzip
import os
import time
import yaml
import warnings
from torchvision import transforms
import numpy as np

def sk_debug():
	debug_idx = [0, 1]
	args = argument_parser()	
	dataloader = SKDataLoader(args)
	model = PointGraphTransformerA()

	## Begins the main operation ##
	for i, (points, labels) in enumerate(dataloader):
		if i==0:
			points = points.to(args.device)
			labels = labels.to(args.device)
			model = model.to(args.device)

			centroids = model(points)
			draw_points_without_labels(centroids[0][800].cpu().numpy())

			return None
			


def s3dis_debug(args):
	dataloader = S3DISDataLoader(args)
	for i, (points, labels) in enumerate(dataloader):
		draw_points_with_labels(points[0][:,:3].numpy(), labels[0].numpy())
		draw_points_with_labels(points[0][:,6:].numpy(), labels[0].numpy())
	return


def s3dis_model_debug(args):
	layers_configs = open_yaml('./configs/DLPT_modelconfig.yaml')['layer_params']['a']
	# with open('./configs/DLPT_modelconfig.yaml') as f:
	# 	layers_configs = yaml.load(f, Loader=yaml.FullLoader)['layer_params']['a']

	model = DLPT(layers_configs)
	model = model.to(args.device)

	dataloader = S3DISDataLoader(args)

	for i, (points, labels)in enumerate(dataloader):
		points, labels = points.to(args.device), labels.to(args.device)
		start_time = time.time()
		pos, feat = model(points)
		print(i, " | Execution time: ", time.time() - start_time)


def modelnet_debug(args):
	configs = open_yaml('./configs/ModelNet40_labelmapconfig.yaml')

	T = transforms.Compose(
	[
		d_utils.PointcloudToTensor(),
		d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
		d_utils.PointcloudScale(),
		d_utils.PointcloudTranslate(),
		d_utils.PointcloudJitter(),
	]
	)

	# dataset = ModelNet40Dataset(num_points=1024, transforms=T)
	dataloader = ModelNet40DataLoader(args, 1024, True, T)
	for i , data in enumerate(dataloader):
		point, label, cluster_idx, downsample_idx = data
		pdb.set_trace()
		# draw_points_without_labels(point.numpy())


def _get_index_file_list(train=True):
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
		# 	z_pad = '0' * (4-len(num))
		# 	num_new = z_pad + num
		# 	new_f = f.split("_")[0] + "_" + num_new + ".msgpack"
		# 	os.rename(os.path.join(index_root, f), os.path.join(index_root, new_f))

		if f[-7:] == 'msgpack':
			if f[:4] == 'clus':
				cluster_filelist.append(os.path.join(index_root, f))
			elif f[:4] == 'down':
				downsample_filelist.append(os.path.join(index_root, f))
	cluster_filelist.sort()
	downsample_filelist.sort()
	return cluster_filelist, downsample_filelist





if __name__ == '__main__':
	args = argument_parser()
	# s3dis_model_debug(args)
	modelnet_debug(args)
	pdb.set_trace()