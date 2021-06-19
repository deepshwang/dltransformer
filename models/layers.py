import torch
import torch.nn as nn
import pgt_ops.pgt_utils as pt_utils
import dgl
import pdb
import faiss
import numpy as np
import math
import sys, os
from utils.visualize import draw_points_without_labels, draw_multiple_points_without_labels


# class DLPTLayer_PreLN(nn.Module):
# 	'''
# 	Decoupled Local Point Transformer Layer
# 	'''
# 	def __init__(self, d_config, downsample_ratio=4, kmeans_ratio=16, expansion_ratio=2, layer_norm=True):
# 		super(DLPTLayer_PreLN, self).__init__()
# 		d_feat_in = d_config[0]
# 		d_pos_embed = d_config[1]
# 		d_feat_hid = d_config[2]
# 		d_feat_embed = d_config[3]
# 		self.DLPTBlock1 = DLPTBlock_PreLN(kmeans_ratio=kmeans_ratio, d_feat=d_feat_in, d_pos_embed=d_pos_embed, d_embed=d_feat_hid, layer_norm=layer_norm)
# 		self.DLPTBlock2 = DLPTBlock_PreLN(kmeans_ratio=expansion_ratio*kmeans_ratio, d_feat=d_feat_hid, d_pos_embed=d_pos_embed, d_embed=d_feat_embed, layer_norm=layer_norm)
# 		self.FPSDownSample = FPS(downsample_ratio=downsample_ratio)


# 	def forward(self, pos, feat, fps_preprocess=None, cluster_batchdict_preprocess_1=None, cluster_batchdict_preprocess_2=None):
# 		feat = self.DLPTBlock1(pos, feat, cluster_batchdict_preprocess_1)
# 		feat = self.DLPTBlock2(pos, feat, cluster_batchdict_preprocess_2)
# 		pos, feat = self.FPSDownSample(pos, feat, fps_preprocess) 
# 		return pos, feat



class DLPTDownLayer(nn.Module):
	'''
	Decoupled Local Point Transformer Layer
	'''
	def __init__(self, d_in, d_out, downsample_ratio, kmeans_ratio, expansion_ratio, layer_norm, dropout_ratio):
		super(DLPTDownLayer, self).__init__()
		self.d_in = d_in
		self.d_out = d_out
		self.DLPTBlock1 = DLPTBlock(kmeans_ratio=kmeans_ratio, 
									d_embed=d_in, 
									layer_norm=layer_norm,
									dropout_ratio=dropout_ratio)
		
		self.DLPTBlock2 = DLPTBlock(kmeans_ratio=expansion_ratio*kmeans_ratio, 
									d_embed=d_in, 
									layer_norm=layer_norm,
									dropout_ratio=dropout_ratio)


		self.trans_down = TransitionDown(downsample_ratio=downsample_ratio, d_in=d_in, d_out=d_out)


	def forward(self, pos, feat, fps_preprocess=None, cluster_batchdict_preprocess_1=None, cluster_batchdict_preprocess_2=None, fpsknn_preprocess=None):
		feat = self.DLPTBlock1(pos, feat, cluster_batchdict_preprocess_1)
		feat = self.DLPTBlock2(pos, feat, cluster_batchdict_preprocess_2)
		pos, feat = self.trans_down(pos, feat, fps_preprocess, fpsknn_preprocess)
		return pos, feat




class DLPTUpLayer(nn.Module):
	'''
	Decoupled Local Point Transformer Layer
	'''
	def __init__(self, d_in, d_out, downsample_ratio, kmeans_ratio, expansion_ratio, layer_norm, dropout_ratio):
		super(DLPTUpLayer, self).__init__()
		self.d_in = d_in
		self.d_out = d_out

		self.trans_up = TransitionUp(d_in=d_in, d_out=d_out)

		self.DLPTBlock1 = DLPTBlock(kmeans_ratio=kmeans_ratio, 
									d_embed=d_out, 
									layer_norm=layer_norm,
									dropout_ratio=dropout_ratio)
		
		self.DLPTBlock2 = DLPTBlock(kmeans_ratio=expansion_ratio*kmeans_ratio, 
									d_embed=d_out, 
									layer_norm=layer_norm,
									dropout_ratio=dropout_ratio)




	def forward(self, pos, feat, pos2, feat2, cluster_batchdict_preprocess_1=None, cluster_batchdict_preprocess_2=None):
		feat = self.trans_up(pos, feat, pos2, feat2)
		feat = self.DLPTBlock1(pos2, feat, cluster_batchdict_preprocess_1)
		feat = self.DLPTBlock2(pos2, feat, cluster_batchdict_preprocess_2)
		return feat



class TransitionDown(nn.Module):
	def __init__(self, downsample_ratio, d_in, d_out):
		super(TransitionDown, self).__init__()
		self.downsample_ratio = downsample_ratio
		self.linear = nn.Linear(d_in, d_out)
		self.bn = nn.BatchNorm1d(d_out)
		self.relu = nn.ReLU()

	def forward(self, pos, feat, fps_preprocess, k_idx):
		B, N, _ = pos.shape
		feat = self.linear(feat)
		feat = torch.transpose(feat, 1, 2).contiguous()
		feat = self.relu(self.bn(feat))
		feat = torch.transpose(feat, 1, 2).contiguous()
		
		if fps_preprocess is not None:
			fps_preprocess = fps_preprocess.view(B, -1)
			pos_downsampled = gather_by_idx(pos, fps_preprocess)
			# feat_downsampled = gather_by_idx(feat, fps_preprocess)
		else:
			fp_idx = pt_utils.farthest_point_sample(pos.contiguous(), int(N/self.downsample_ratio))
			pos_downsampled = gather_by_idx(pos, fp_idx)
			# feat_downsampled = gather_by_idx(feat, fp_idx)

		feat_downsampled = []
		if k_idx is not None:
			for b in range(B):
				p = pos[b, :]
				f = feat[b,:]
				idx = k_idx[b]
				feat_knn = f[torch.from_numpy(idx)]

				# Debugging whether knn index is corrent => IT IS CORRECT
				# p_knns = p[torch.from_numpy(idx)]
				# for p_knn in p_knns:
				# 	draw_multiple_points_without_labels([p.cpu(), p_knn.cpu()])
				# 	pdb.set_trace()

				feat_maxed = torch.max(feat_knn, dim=1).values
				feat_downsampled.append(feat_maxed)

		else:
			for b in range(B):
				p = pos[b,:]
				f = feat[b,:]
				p_downsampled = pos_downsampled[b,:]

				k_idx = faissKNN(index=p, query=p_downsampled, k=16)
				# feat_downsampled.append(torch.mean(f[torch.from_numpy(k_idx)], dim=1))

				feat_downsampled.append(torch.max(f[torch.from_numpy(k_idx)], dim=1).values)


		feat_downsampled = torch.stack(feat_downsampled)

		return pos_downsampled, feat_downsampled


class TransitionUp(nn.Module):

	def __init__(self, d_in, d_out):
		super().__init__()

		self.up_mlp = nn.Sequential(
			nn.Conv1d(d_in, d_out, kernel_size=1, bias=False),
			nn.BatchNorm1d(d_out),
			nn.ReLU(True)
		)
		self.lateral_mlp = nn.Sequential(
			nn.Conv1d(d_out, d_out, kernel_size=1, bias=False),
			nn.BatchNorm1d(d_out),
			nn.ReLU(True)
		)


	def forward(self, x1, p1, x2, p2):
		"""
			x1: (B, N, d_in) torch.Tensor
			p1: (B, N, 3) torch.Tensor
			x2: (B, M, d_out) torch.Tensor
			p2: (B, M, 3) torch.Tensor
			Note that N is smaller than M because this module upsamples features.
		"""
		x1 = self.up_mlp(x1.transpose(1, 2).contiguous())
		dist, idx= pt_utils.three_nn(p2, p1)
		dist_recip = 1.0 / (dist + 1e-8)
		norm = torch.sum(dist_recip, dim=2, keepdim=True)
		weight = dist_recip / norm
		interpolated_feats = pt_utils.three_interpolate(
			x1, idx, weight
		)
		x2 = self.lateral_mlp(x2.transpose(1, 2).contiguous())
		y = interpolated_feats + x2
		return y.transpose(1, 2).contiguous(), p2





class DLPTBlock(nn.Module):
	'''
	Decoupled Local Point Transformer Block
	'''
	def __init__(self, kmeans_ratio, d_embed, layer_norm, dropout_ratio):
		super(DLPTBlock, self).__init__()
		self.kmeans_ratio = kmeans_ratio
		self.d_embed = d_embed
		self.lpe = LPEBlock(d_embed=d_embed)
		self.dlsa = DLSABlock(d_embed=d_embed)
		self.layer_norm = None
		if layer_norm:
			self.layer_norm = layer_norm
			self.ln1 = nn.LayerNorm(self.d_embed)
			self.ln2 = nn.LayerNorm(self.d_embed)
		if dropout_ratio is not None:
			self.ff = nn.Sequential(nn.Linear(d_embed, d_embed*4),
									nn.ReLU(),
									nn.Dropout(dropout_ratio),
									nn.Linear(d_embed*4, d_embed))
		else:
			self.ff = nn.Sequential(nn.Linear(d_embed, d_embed*4),
									nn.ReLU(),
									nn.Linear(d_embed*4, d_embed))

		if dropout_ratio is not None:
			self.dropout1 = nn.Dropout(dropout_ratio)
			self.dropout2 = nn.Dropout(dropout_ratio)
		else:
			self.dropout1 = self.dropout2 = None


	def forward(self, pos, feat, cluster_batchdict):
		# [1] Decoupled Local Self Attention
		if cluster_batchdict is None:
			cluster_batchdict = get_cluster_idxes(pos, kmeans_ratio=self.kmeans_ratio)

		h_pos, h_geo = self.lpe(pos, feat, cluster_batchdict)

		feat_out = self.dlsa(h_pos, h_geo, cluster_batchdict)

		if self.dropout1 is not None:
			feat_out = self.dropout1(feat_out)

		# [2] Skip connection + Layer Norm
		feat_out = h_pos + feat_out
		if self.layer_norm:
			feat_out = self.ln1(feat_out) 
		
		# [3] Feed Forward + dropout & Skip connection + Layer Norm
		final_out = self.ff(feat_out)
		if self.dropout2 is not None:
			final_out = self.dropout2(final_out)
		final_out = final_out + feat_out
		if self.layer_norm is not None:
			final_out = self.ln2(final_out)

		return feat_out


# class DLPTBlock_PreLN(nn.Module):
# 	'''
# 	Decoupled Local Point Transformer Block with Pre Layer-Normalization
# 	'''
# 	def __init__(self, kmeans_ratio=16, d_feat=3, d_pos_embed=10, d_embed=32, layer_norm=True):
# 		super(DLPTBlock_PreLN, self).__init__()
# 		self.kmeans_ratio = kmeans_ratio
# 		self.d_pos = 3
# 		self.d_pos_embed = d_pos_embed
# 		self.d_embed = d_embed
# 		self.lpe = LPEBlock(d_feat=d_feat, d_pos_embed=d_pos_embed, d_embed=d_embed)
# 		self.dlsa = DLSABlock(d_embed=d_embed)
# 		self.layer_norm = None
# 		if layer_norm:
# 			self.layer_norm = layer_norm
# 			self.ln11 = nn.LayerNorm(self.d_embed)
# 			self.ln12 = nn.LayerNorm(self.d_embed)
# 			self.ln2 = nn.LayerNorm(self.d_embed)
# 		self.ff = nn.Sequential(nn.Linear(d_embed, d_embed*4),
# 								nn.ReLU(),
# 								nn.Linear(d_embed*4, d_embed))



# 	def forward(self, pos, feat, cluster_batchdict):
# 		# [1] Decoupled Local Self Attention
# 		if cluster_batchdict is None:
# 			cluster_batchdict = get_cluster_idxes(pos, kmeans_ratio=self.kmeans_ratio)

# 		h_pos, h_geo = self.lpe(pos, feat, cluster_batchdict)
# 		try:
# 			h_pos_dlsa = self.ln11(h_pos)
# 			h_geo_dlsa = self.ln12(h_geo)
# 		except torch.nn.modules.module.ModuleAttributeError:
# 			print("PreLN requires Layer norm. Check if you set layer_norm as false in model configuration yaml")
# 		feat_out = self.dlsa(h_pos_dlsa, h_geo_dlsa, cluster_batchdict)
		
# 		# [2] Skip connection 
# 		feat_out = h_pos + feat_out
 
		
# 		# [3] Feed Forward & Skip connection + Layer Norm
# 		final_out = self.ln2(self.ff(feat_out))
# 		final_out = final_out + feat_out

# 		return feat_out
		


class DLSABlock(nn.Module):
	'''
	Decoupled Local Self-Attention Block
	TODO: Multi-head
	'''
	def __init__(self, d_embed=32):
		super(DLSABlock, self).__init__()
		self.project_q = nn.Linear(d_embed, d_embed, bias=False)
		self.project_k = nn.Linear(d_embed, d_embed, bias=False)
		self.project_v = nn.Linear(d_embed, d_embed, bias=False)
		self.d_embed = d_embed
		self.linear_out = nn.Linear(d_embed, d_embed)


	def forward(self, h_pos, h_geo, kmeans_idx_dict_batchlist):
		Q = self.project_q(h_geo)
		K = self.project_k(h_geo)
		V = self.project_v(h_pos)

		attn = torch.empty(V.shape).to(V.device)
		for b, kmeans_idx_dict in enumerate(kmeans_idx_dict_batchlist):
			for c, (_, cluster_idx) in enumerate(kmeans_idx_dict.items()):
				Q_c = Q[b][cluster_idx] / math.sqrt(self.d_embed)
				K_c = K[b][cluster_idx]
				V_c = V[b][cluster_idx]
				attn_c = nn.functional.softmax(torch.mm(Q_c, K_c.transpose(1, 0)), dim=1)
				out = torch.mm(attn_c, V_c)
				attn[b][cluster_idx] = out
		attn = self.linear_out(attn)
		return attn




class LPEBlock(nn.Module):
	'''
	Local Position Embedding Block
	'''
	def __init__(self, d_embed):
		super(LPEBlock, self).__init__()
		self.kmeans_ratio = 8
		self.d_embed = d_embed
		self.d_pos = 3
		self.mlp_1a = self._make_embedding_layers(4, d_embed)
		# self.mlp_1b = self._make_embedding_layers(d_pos_embed + d_feat, d_embed)
		self.mlp_2a = self._make_embedding_layers(6, d_embed)
		# self.mlp_2b = self._make_embedding_layers(d_pos_embed + d_feat, d_embed)


	def forward(self, pos, feat, kmeans_idx_dict_batchlist):
		'''
		Args:
			x ()

		Returns:

		'''
		# pos = x[:, :, :3]
		# feat = x[:, :, 3:6]
		B, N, _ = feat.shape
		h_pos = torch.empty(B, N, self.d_embed).to(feat.device)
		h_geo = torch.empty(B, N, self.d_embed).to(feat.device)
		
		# Clusterize points with k-means clustering algorithm
		# kmeans_idx_dict_batchlist = get_cluster_idxes(pos, kmeans_ratio=self.kmeans_ratio)
		
		# Iterate for every cluster 
		for b, kmeans_idx_dict in enumerate(kmeans_idx_dict_batchlist):
			for c, (_, cluster_idx) in enumerate(kmeans_idx_dict.items()):
				## [1] Relative Position Embedding (RPE)
				p_c = pos[b][cluster_idx]
				f_c = feat[b][cluster_idx]

				h_pos_c, h_geo_c = self.embed_LPE(p_c, f_c)
				
				h_pos[b][cluster_idx] = h_pos_c
				h_geo[b][cluster_idx] = h_geo_c
 

		return h_pos, h_geo


	def _make_embedding_layers(self, d_in, d_embed, layer_norm=False, relu=True):
		modules=[]
		modules.append(nn.Linear(d_in, d_embed))
		if layer_norm:
			modules.append(nn.LayerNorm(d_embed))
		if relu:
			modules.append(nn.ReLU())
			modules.append(nn.Linear(d_embed, d_embed))

		sequential = nn.Sequential(*modules)
		return sequential


	def embed_LPE(self, p, f):
		cog = torch.mean(p, dim=0, keepdim=True)
		local_p = p - cog
		n = torch.norm(local_p, dim=1, keepdim=True)
		# r = self.mlp_1a(torch.cat((local_p, n), dim=1))
		embed_pos = self.mlp_1a(torch.cat((local_p, n), dim=1))
		
		## [1] Relative Position Embedding
		# h_pos = self.mlp_1b(torch.cat((r, f), dim=1))
		h_pos = f + embed_pos


		## [2] Relative Geometry Embedding 
		avg = torch.mean(local_p, dim=0, keepdim=True)
		embed_geo = self.mlp_2a(torch.cat((avg.expand(local_p.shape), local_p), dim=1))
		# h_geo = self.mlp_2b(torch.cat((r_hat, f), dim=1))
		h_geo = f + embed_geo

		return h_pos, h_geo


def faissKMeans(pos, kmeans_ratio):
	'''
	Argument:
		pos (N, C): Non-batched point position tensor
		kmeans_ratio: number of cluster is determined as int(N/kmeans_ratio)
	
	Return:
		idxs (N, 1): Every point is assigned with a cluster index number
	'''

	N, C = pos.shape
	pos = pos.cpu().numpy().astype(np.float32)
	faiss_kmeans = faiss.Kmeans(d=C, k=int(N/kmeans_ratio), niter=150, verbose=False, gpu=True)
	faiss_kmeans.min_points_per_centroid=1
	faiss_kmeans.train(pos)
	dists, idxs = faiss_kmeans.index.search(pos, 1)
	return idxs

def get_cluster_idxes(pos, kmeans_ratio):
	B, N, C = pos.shape
	device= pos.device
	cluster_idx_dict_list = []
	for b in range(B):
		cluster_assignment_idxs = faissKMeans(pos[b], kmeans_ratio)
		cluster_idx_dict={}
		for i, cluster_idx in enumerate(np.squeeze(cluster_assignment_idxs)):
			cluster_idx = int(cluster_idx)
			if cluster_idx not in cluster_idx_dict.keys():
				cluster_idx_dict[cluster_idx] = [i]
			else:
				cluster_idx_dict[cluster_idx].append(i)

		cluster_idx_dict_list.append(cluster_idx_dict)

	return cluster_idx_dict_list

def faissKNN(index, query, k):
	index = index.cpu().numpy().copy(order='C')

	query = query.cpu().numpy()
	
	N = query.shape[0]

	faiss_index = faiss.IndexFlatL2(index.shape[1])
	faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
	faiss_index.add(index)

	_, k_idx = faiss_index.search(query, k)


	# neighbor_idx = np.squeeze(k_idx.reshape(1,-1)).astype(np.int64)
	# node_idx = np.concatenate([(np.ones(k) * (i+1) - 1).astype(np.int64) for i in range(N)], axis=0)

	return k_idx

	# end_time = time.time()
	# print("Time (sec) to conduct KNN per pc: ", end_time - start_time)

def gather_by_idx(db, q_idx):
	db_flipped = torch.einsum("ijk->ikj", db).contiguous()
	db = pt_utils.gather_operation(db_flipped, q_idx)
	db = torch.einsum("ijk->ikj", db).contiguous()

	return db