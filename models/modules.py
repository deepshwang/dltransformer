import torch
import torch.nn as nn
import pgt_ops.pgt_utils as pt_utils

from models.layers import DLPTLayer
import pdb
from utils.visualize import draw_points_with_labels, draw_points_without_labels


class DLPTNet_cls(nn.Module):
	'''
	Decoupled Local Point Transformer Model for classification
	'''
	def __init__(self, layer_configs, c):
		super(DLPTNet_cls, self).__init__()
		self.ds_ratio = layer_configs['ds_ratio']
		self.k = layer_configs['k']
		self.expansion_ratio = layer_configs['expansion_ratio']

		self.ds1 = DLPTLayer(3, 16, 32, self.ds_ratio, self.k, self.expansion_ratio)
		self.ds2 = DLPTLayer(32, 32, 64, self.ds_ratio, self.k, self.expansion_ratio)
		self.ds3 = DLPTLayer(64, 64, 128, self.ds_ratio, self.k, self.expansion_ratio)
		self.ds4 = DLPTLayer(128, 128, 256, self.ds_ratio, self.k, self.expansion_ratio)
		self.classifier = nn.Sequential(nn.Linear(256, 64),
										nn.ReLU(),
										nn.Linear(64, 40),
										nn.Sigmoid())

		self._reset_parameters()



	def forward(self, x, c_pre=None, ds_pre=None):
		if c_pre is None:
			c_pre_11 = c_pre_12 = c_pre_21 = c_pre_22 = c_pre_31 = c_pre_32 = c_pre_41 = c_pre_42 = None
		else:
			c_pre_11 = c_pre[0]
			c_pre_12 = c_pre[1]
			c_pre_21 = c_pre[2]
			c_pre_22 = c_pre[3]
			c_pre_31 = c_pre[4]
			c_pre_32 = c_pre[5]
			c_pre_41 = c_pre[6]
			c_pre_42 = c_pre[7]


		if ds_pre is None:
			ds_pre_1 = None
			ds_pre_2 = None
			ds_pre_3 = None
			ds_pre_4 = None

		else:
			ds_pre_1 = ds_pre[0]
			ds_pre_2 = ds_pre[1]
			ds_pre_3 = ds_pre[2]
			ds_pre_4 = ds_pre[3]


		pos = x[:, :, :3]
		feat = x[:, :, 3:6]
		pos_ds, feat_ds = self.ds1(pos, feat, ds_pre_1, c_pre_11, c_pre_12)
		pos_ds, feat_ds = self.ds2(pos_ds, feat_ds, ds_pre_2, c_pre_21, c_pre_22)
		pos_ds, feat_ds = self.ds3(pos_ds, feat_ds, ds_pre_3, c_pre_31, c_pre_32)
		pos_ds, feat_ds = self.ds4(pos_ds, feat_ds, ds_pre_4, c_pre_41, c_pre_42)
		out = torch.mean(feat_ds, dim=1)
		out = self.classifier(out)
		return out

	def _reset_parameters(self):
		"""Xavier Initialization of parameters"""
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)