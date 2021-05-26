import torch
import torch.nn as nn
import pgt_ops.pgt_utils as pt_utils
import dgl
import pdb
import faiss
import numpy as np
import math
import sys, os


class DLPTLayer(nn.Module):
	'''
	Decoupled Local Point Transformer Layer
	'''
	def __init__(self, d_feat_in=3, d_feat_hid=32, d_feat_out=64, downsample_ratio=4, kmeans_ratio=16, expansion_ratio=2):
		super(DLPTLayer, self).__init__()
		self.DLPTBlock1 = DLPTBlock(kmeans_ratio=kmeans_ratio, d_feat=d_feat_in, d_pos_embed=d_feat_in, d_embed=d_feat_hid)
		self.DLPTBlock2 = DLPTBlock(kmeans_ratio=expansion_ratio*kmeans_ratio, d_feat=d_feat_hid, d_pos_embed=d_feat_hid, d_embed=d_feat_out)
		self.FPSDownSample = FPS(downsample_ratio=downsample_ratio)


	def forward(self, pos, feat, fps_preprocess=None, cluster_batchdict_preprocess_1=None, cluster_batchdict_preprocess_2=None):
		feat = self.DLPTBlock1(pos, feat, cluster_batchdict_preprocess_1)
		feat = self.DLPTBlock2(pos, feat, cluster_batchdict_preprocess_2)
		pos, feat = self.FPSDownSample(pos, feat, fps_preprocess)
		return pos, feat



class FPS(nn.Module):
	def __init__(self, downsample_ratio=4):
		super(FPS, self).__init__()
		self.downsample_ratio = downsample_ratio


	def forward(self, pos, feat, fps_preprocess):
		B, N, _ = pos.shape
		if fps_preprocess is not None:
			fps_preprocess = torch.squeeze(fps_preprocess)
			pos_downsampled = self.gather_by_idx(pos, fps_preprocess)
			feat_downsampled = self.gather_by_idx(feat, fps_preprocess)
		else:
			fp_idx = pt_utils.farthest_point_sample(pos.contiguous(), int(N/self.downsample_ratio))
			pos_downsampled = self.gather_by_idx(pos, fp_idx)
			feat_downsampled = self.gather_by_idx(feat, fp_idx)
		return pos_downsampled, feat_downsampled


	def gather_by_idx(self, db, q_idx):
		db_flipped = torch.einsum("ijk->ikj", db).contiguous()
		db = pt_utils.gather_operation(db_flipped, q_idx)
		db = torch.einsum("ijk->ikj", db).contiguous()

		return db



class DLPTBlock(nn.Module):
	'''
	Decoupled Local Point Transformer Block
	'''
	def __init__(self, kmeans_ratio=16, d_feat=3, d_pos_embed=10, d_embed=32):
		super(DLPTBlock, self).__init__()
		self.kmeans_ratio = kmeans_ratio
		self.d_pos = 3
		self.d_pos_embed = d_pos_embed
		self.d_embed = d_embed
		self.lpe = LPEBlock(d_feat=d_feat, d_pos_embed=d_pos_embed, d_embed=d_embed)
		self.dlsa = DLSABlock(d_embed=d_embed)
		self.ln = nn.LayerNorm(self.d_embed)
		self.ff = nn.Linear(d_embed, d_embed)


	def forward(self, pos, feat, cluster_batchdict):
		# [1] Decoupled Local Self Attention
		if cluster_batchdict is None:
			cluster_batchdict = get_cluster_idxes(pos, kmeans_ratio=self.kmeans_ratio)
		h_pos, h_geo = self.lpe(pos, feat, cluster_batchdict)
		feat_out = self.dlsa(h_pos, h_geo, cluster_batchdict)
		
		# [2] Skip connection + Layer Norm
		feat_out = h_pos + feat_out

		feat_out = self.ln(feat_out) 
		
		# [3] Feed Forward & Skip connection + Layer Norm
		final_out = self.ff(feat_out)
		final_out = final_out + feat_out
		final_out = self.ln(final_out)

		return feat_out
		


class DLSABlock(nn.Module):
	'''
	Decoupled Local Self-Attention Block
	TODO: Multi-head
	'''
	def __init__(self, d_embed=32):
		super(DLSABlock, self).__init__()
		self.project_q = nn.Linear(d_embed, d_embed)
		self.project_k = nn.Linear(d_embed, d_embed)
		self.project_v = nn.Linear(d_embed, d_embed)
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
	def __init__(self, d_feat, d_pos_embed, d_embed):
		super(LPEBlock, self).__init__()
		self.kmeans_ratio = 8
		self.d_pos = 3
		self.d_feat = d_feat
		self.d_pos_embed = d_pos_embed
		self.d_embed = d_embed
		self.mlp_1a = self._make_embedding_layers(self.d_pos+1, d_pos_embed)
		self.mlp_1b = self._make_embedding_layers(d_pos_embed + d_feat, d_embed)
		self.mlp_2a = self._make_embedding_layers(self.d_pos*2, d_pos_embed)
		self.mlp_2b = self._make_embedding_layers(d_pos_embed + d_feat, d_embed)


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


	def _make_embedding_layers(self, d_in, d_out, layer_norm=False, relu=False):
		modules=[]
		modules.append(nn.Linear(d_in, d_out))
		if layer_norm:
			modules.append(nn.LayerNorm(d_out))
		if relu:
			modules.append(nn.ReLU())

		sequential = nn.Sequential(*modules)
		return sequential


	def embed_LPE(self, p, f):
		cog = torch.mean(p, dim=0, keepdim=True)
		local_p = p - cog
		n = torch.norm(local_p, dim=1, keepdim=True)
		r = self.mlp_1a(torch.cat((local_p, n), dim=1))
		
		## [1] Relative Position Embedding
		h_pos = self.mlp_1b(torch.cat((r, f), dim=1))


		## [2] Relative Geometry Embedding 
		avg = torch.mean(local_p, dim=0, keepdim=True)
		r_hat = self.mlp_2a(torch.cat((avg.expand(local_p.shape), local_p), dim=1))
		h_geo = self.mlp_2b(torch.cat((r_hat, f), dim=1))
		return h_pos, h_geo


def faissKMeans(pos, kmeans_ratio):
	N, C = pos.shape
	pos = pos.cpu().numpy().astype(np.float32)
	with HiddenPrints():
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

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


