import torch
import torch.nn as nn
import pgt_ops.pgt_utils as pt_utils

from models.layers import DLPTDownLayer, DLPTUpLayer
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
		self.d_config = layer_configs['d_config']
		self.layer_norm = layer_configs['layer_norm']
		self.dropout_ratio = layer_configs['dropout_ratio']
		self.ff = nn.Linear(self.d_config[0], self.d_config[1])
		self.ds1 = DLPTDownLayer(self.d_config[1], self.d_config[2], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.ds2 = DLPTDownLayer(self.d_config[2], self.d_config[3], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.ds3 = DLPTDownLayer(self.d_config[3], self.d_config[4], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.ds4 = DLPTDownLayer(self.d_config[4], self.d_config[5], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.classifier = nn.Sequential(nn.Linear(self.d_config[-1], 64),
										nn.ReLU(),
										nn.Linear(64, 40))
		self._reset_parameters()



	def forward(self, x, c_pre=None, ds_pre=None, fpsknn_pre=None):
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


		if fpsknn_pre is None:
			fpsknn_pre_1 = fpsknn_pre_2 = fpsknn_pre_3 = fpsknn_pre_4 = None
		else:
			fpsknn_pre_1 = fpsknn_pre[0]
			fpsknn_pre_2 = fpsknn_pre[1]
			fpsknn_pre_3 = fpsknn_pre[2]
			fpsknn_pre_4 = fpsknn_pre[3]


		pos = x[:, :, :3]
		feat = x[:, :, 3:6]
		feat = self.ff(feat)
		pos_ds, feat_ds = self.ds1(pos, feat, ds_pre_1, c_pre_11, c_pre_12, fpsknn_pre_1)
		pos_ds, feat_ds = self.ds2(pos_ds, feat_ds, ds_pre_2, c_pre_21, c_pre_22, fpsknn_pre_2)
		pos_ds, feat_ds = self.ds3(pos_ds, feat_ds, ds_pre_3, c_pre_31, c_pre_32, fpsknn_pre_3)
		pos_ds, feat_ds = self.ds4(pos_ds, feat_ds, ds_pre_4, c_pre_41, c_pre_42, fpsknn_pre_4)
		out = torch.mean(feat_ds, dim=1)
		out = self.classifier(out)
		return out

	def _reset_parameters(self):
		"""Xavier Initialization of parameters"""
		torch.manual_seed(77)
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)


class DLPTNet_seg(nn.Module):
	'''
	Decoupled Local Point Transformer Model for classification
	'''
	def __init__(self, layer_configs, c):
		super(DLPTNet_cls, self).__init__()
		self.ds_ratio = layer_configs['ds_ratio']
		self.k = layer_configs['k']
		self.expansion_ratio = layer_configs['expansion_ratio']
		self.d_config = layer_configs['d_config']
		self.layer_norm = layer_configs['layer_norm']
		self.dropout_ratio = layer_configs['dropout_ratio']
		self.ff = nn.Sequential(
			nn.Conv1d(self.d_config[0], self.d_config[1], kernel_size=1, bias=False),
			nn.BatchNorm1d(d_config[5]),
			nn.ReLU(True))

		self.ds12 = DLPTDownLayer(self.d_config[1], self.d_config[2], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.ds23 = DLPTDownLayer(self.d_config[2], self.d_config[3], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.ds34 = DLPTDownLayer(self.d_config[3], self.d_config[4], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.ds45 = DLPTDownLayer(self.d_config[4], self.d_config[5], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		
		self.intermediate = nn.Sequential(
			nn.Conv1d(self.d_config[5], self.d_config[5], kernel_size=1, bias=False),
			nn.BatchNorm1d(d_config[5]),
			nn.ReLU(True))
		
		self.us54 = DLPTUpLayer(self.d_config[5], self.d_config[4], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.us43 = DLPTUpLayer(self.d_config[4], self.d_config[3], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.us32 = DLPTUpLayer(self.d_config[3], self.d_config[2], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)
		self.us21 = DLPTUpLayer(self.d_config[2], self.d_config[1], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm, dropout_ratio=self.dropout_ratio)

		self.classifier = nn.Sequential(
			nn.Conv1d(self.d_config[1], self.d_config[1], kernel_size=1, bias=False),
			nn.BatchNorm1d(self.d_config[1]),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv1d(self.d_config[1], c, kernel_size=1),
		)

		self._reset_parameters()



	def forward(self, x, c_pre=None, ds_pre=None, fpsknn_pre=None):
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


		if fpsknn_pre is None:
			fpsknn_pre_1 = fpsknn_pre_2 = fpsknn_pre_3 = fpsknn_pre_4 = None
		else:
			fpsknn_pre_1 = fpsknn_pre[0]
			fpsknn_pre_2 = fpsknn_pre[1]
			fpsknn_pre_3 = fpsknn_pre[2]
			fpsknn_pre_4 = fpsknn_pre[3]


		pos1 = x[:, :, :3]
		feat = x[:, :, 3:6]
		feat1 = self.ff(feat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
		pos2, feat2 = self.ds1(pos1, feat1, ds_pre_1, c_pre_11, c_pre_12, fpsknn_pre_1)
		pos3, feat3 = self.ds2(pos2, feat2, ds_pre_2, c_pre_21, c_pre_22, fpsknn_pre_2)
		pos4, feat4 = self.ds3(pos3, feat3, ds_pre_3, c_pre_31, c_pre_32, fpsknn_pre_3)
		pos5, feat5 = self.ds4(pos4, feat4, ds_pre_4, c_pre_41, c_pre_42, fpsknn_pre_4)

		feat5 = self.intermediate(feat5.transpose(1, 2).contiguous()).trasnpose(1, 2).contiguous()

		feat4 = self.us54(pos5, feat5, pos4, feat4, c_pre_41, c_pre_42)
		feat3 = self.us43(pos4, feat4, pos3, feat3, c_pre_31, c_pre_32)
		feat2 = self.us32(pos3, feat3, pos2, feat2, c_pre_21, c_pre_22)
		feat1 = self.us21(pos2, feat2, pos1, feat1, c_pre_11, c_pre_12)

		out = self.classifier(feat1.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

		return out

	def _reset_parameters(self):
		"""Xavier Initialization of parameters"""
		torch.manual_seed(77)
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)



# class DLPTNet_PreLN_cls(nn.Module):
# 	'''
# 	Decoupled Local Point Transformer Model for classification
# 	'''
# 	def __init__(self, layer_configs, c):
# 		super(DLPTNet_PreLN_cls, self).__init__()
# 		self.ds_ratio = layer_configs['ds_ratio']
# 		self.k = layer_configs['k']
# 		self.expansion_ratio = layer_configs['expansion_ratio']
# 		self.d_config = layer_configs['d_config']
# 		self.layer_norm = layer_configs['layer_norm']
# 		self.ff = nn.Linear(3, self.d_config[0])
# 		self.ds1 = DLPTLayer_PreLN(self.d_config[1], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm)
# 		self.ds2 = DLPTLayer_PreLN(self.d_config[2], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm)
# 		self.ds3 = DLPTLayer_PreLN(self.d_config[3], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm)
# 		self.ds4 = DLPTLayer_PreLN(self.d_config[4], self.ds_ratio, self.k, self.expansion_ratio, layer_norm=self.layer_norm)
# 		self.classifier = nn.Sequential(nn.Linear(self.d_config[-1][-1], 64),
# 										nn.ReLU(),
# 										nn.Linear(64, 40),
# 										nn.Sigmoid())

# 		self._reset_parameters()



# 	def forward(self, x, c_pre=None, ds_pre=None, fpsknn_pre=None):
# 		if c_pre is None:
# 			c_pre_11 = c_pre_12 = c_pre_21 = c_pre_22 = c_pre_31 = c_pre_32 = c_pre_41 = c_pre_42 = None
# 		else:
# 			c_pre_11 = c_pre[0]
# 			c_pre_12 = c_pre[1]
# 			c_pre_21 = c_pre[2]
# 			c_pre_22 = c_pre[3]
# 			c_pre_31 = c_pre[4]
# 			c_pre_32 = c_pre[5]
# 			c_pre_41 = c_pre[6]
# 			c_pre_42 = c_pre[7]


# 		if ds_pre is None:
# 			ds_pre_1 = None
# 			ds_pre_2 = None
# 			ds_pre_3 = None
# 			ds_pre_4 = None
# 		else:
# 			ds_pre_1 = ds_pre[0]
# 			ds_pre_2 = ds_pre[1]
# 			ds_pre_3 = ds_pre[2]
# 			ds_pre_4 = ds_pre[3]


# 		if fpsknn_pre is None:
# 			fpsknn_pre_1 = fpsknn_pre_2 = fpsknn_pre_3 = fpsknn_pre_4 = None
# 		else:
# 			fpsknn_pre_1 = fpsknn_pre[0]
# 			fpsknn_pre_2 = fpsknn_pre[1]
# 			fpsknn_pre_3 = fpsknn_pre[2]
# 			fpsknn_pre_4 = fpsknn_pre[3]


# 		pos = x[:, :, :3]
# 		feat = x[:, :, 3:6]
# 		feat = self.ff(feat)
# 		pdb.set_trace()
# 		pos_ds, feat_ds = self.ds1(pos, feat, ds_pre_1, c_pre_11, c_pre_12)
# 		pos_ds, feat_ds = self.ds2(pos_ds, feat_ds, ds_pre_2, c_pre_21, c_pre_22)
# 		pos_ds, feat_ds = self.ds3(pos_ds, feat_ds, ds_pre_3, c_pre_31, c_pre_32)
# 		pos_ds, feat_ds = self.ds4(pos_ds, feat_ds, ds_pre_4, c_pre_41, c_pre_42)
# 		out = torch.mean(feat_ds, dim=1)
# 		out = self.classifier(out)
# 		return out

	def _reset_parameters(self):
		"""Xavier Initialization of parameters"""
		torch.manual_seed(77)
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)