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

	# args, num_points, shuffle, train, transforms
	dataloader = ModelNet40DataLoader(args, 1024, True, True, T)
	for i , data in enumerate(dataloader):
		point, label, cluster_idx, downsample_idx = data
		pdb.set_trace()
		point, label, cluster_idx, downsample_idx = point[0], label[0], cluster_idx[0], downsample_idx[0]
		pdb.set_trace()
		# draw_points_without_labels(point.numpy())


def cluster_visualize_debug(args):
	config = open_yaml('./configs/ModelNet40_labelmapconfig.yaml')
	
	T = transforms.Compose(
	[
		d_utils.PointcloudToTensor(),
		# d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
		# d_utils.PointcloudScale(),
		# d_utils.PointcloudTranslate(),
		d_utils.PointcloudJitter(),
	]
	)

	dataset = ModelNet40Dataset(1024, transforms=T, train=True)

	for i in range(len(dataset)):
		point, label, cluster_idx, downsample_idx = dataset[i]
		downsample_idx = downsample_idx.numpy().tolist()[0]

		N = point.shape[0]
		cluster_a = cluster_idx[0]
		cluster_b = cluster_idx[1]

		cluster_a_idx = label_cluster_dict(cluster_idx[0], N)
		cluster_b_idx = label_cluster_dict(cluster_idx[1], N)

		draw_points_with_labels(point[:,:3].numpy(), cluster_a_idx)
		draw_points_with_labels(point[:,:3].numpy(), cluster_b_idx)
		point = point[downsample_idx]
		N = point.shape[0]
		cluster_c_idx = label_cluster_dict(cluster_idx[2], N)
		cluster_d_idx = label_cluster_dict(cluster_idx[3], N)
		draw_points_with_labels(point[:,:3].numpy(), cluster_c_idx)
		draw_points_with_labels(point[:,:3].numpy(), cluster_d_idx)


		point = point[:64, :]
		N = point.shape[0]
		cluster_e_idx = label_cluster_dict(cluster_idx[4], N)
		cluster_f_idx = label_cluster_dict(cluster_idx[5], N)
		draw_points_with_labels(point[:,:3].numpy(), cluster_e_idx)
		draw_points_with_labels(point[:,:3].numpy(), cluster_f_idx)

		point = point[:16, :]
		N = point.shape[0]
		cluster_g_idx = label_cluster_dict(cluster_idx[6], N)
		cluster_h_idx = label_cluster_dict(cluster_idx[7], N)
		draw_points_without_labels(point[:,:3].numpy())
		draw_points_without_labels(point[:,:3].numpy())

		pdb.set_trace()

def label_cluster_dict(cluster_dict, N):
	cluster_label = np.zeros(N)
	for key, value in cluster_dict.items():
		cluster_label[value] = key
	return cluster_label


if __name__ == '__main__':
	args = argument_parser()
	# s3dis_model_debug(args)
	cluster_visualize_debug(args)
	pdb.set_trace()