import torch
import numpy as np
import dgl
import pdb
import faiss
# from knn_cuda import KNN
import time
from dgl.data import DGLDataset
from sklearn.metrics import pairwise_distances_argmin_min

from utils.util import load_pickle



class sKGraphDataset(torch.utils.data.Dataset):
	def __init__(self, pc_dataset, dataset_config, preprocess_device=None, preprocess=False):
		self.pc_dataset = pc_dataset
		self.preprocess = preprocess
		self.preprocessed_path = [self._get_preprocessed_path(f) for f in pc_dataset.im_idx]
		if preprocess:
			self.device = preprocess_device
		self.n_centroids = int(dataset_config['graph_dataset_params']['n_centroids'])
		self.k_rsc_g = int(dataset_config['graph_dataset_params']['k_rsc_g'])
		self.k_rsc_rsc = int(dataset_config['graph_dataset_params']['k_rsc_rsc'])


	def __len__(self):
		return len(self.pc_dataset)


	def __getitem__(self, index):
		if self.preprocess:
			points, labels = self.pc_dataset[index]
			g = self._create_graph(points, labels)
		else: 
			g = load_pickle(self.preprocessed_path[index])
			_, labels = self.pc_dataset[index]
		return g, labels


	def collate_fn(self, samples):
		'''
		samples: list of pairs of (graph)
		'''
		graphs, labels = map(list, zip(*samples))
		labels = torch.squeeze(torch.cat(labels).long())
		batched_graph = dgl.batch(graphs)

		return batched_graph, labels


	def _create_graph(self, points, labels):
		N, _ = points.shape

		# [1] Randomly select self.n_centroid number of centroids & remainders are members
		RSCs, members = self.getCentroids(points)

		# [2] Find k_sc_g - nearest neighbors of centroids w.r.t global points
		RSCtoG_idx, GfromRSC_idx = self._faissKNNEdge(k=self.k_rsc_g, index=points, query=RSCs)

		# [3] Find k_sc_sc - nearest neighbors of SCs w.r.t itself (SCs).
		RSCtoRSC_idx, RSCfromRSC_idx = self._faissKNNEdge(k=self.k_rsc_rsc, index=RSCs)


		# [4] Create heterogeneous graph
		# (src, data, dst)
		g = dgl.heterograph(
			{('G', 'e_RSC_G', 'RSC'): (GfromRSC_idx, RSCtoG_idx),
			('RSC', 'e_RSC_RSC', 'RSC'): (RSCfromRSC_idx, RSCtoRSC_idx)
			})


		g.ndata['RSC'] = {'RSC': RSCs}
		g.ndata['G'] = {'G': points}


		###### DEPRECATED: FULLY CONNECTED GRAPH ######

		# Initialize graph with K number of edges
		# g = self._initialize_graph(points)

		# Add node feature #1: Original point
		# g.ndata['h_p'] = points

		# Add edge feature #1: Relative positions between nodes
		# g = g.to(self.device)
		# g.apply_edges(dgl.function.v_sub_u('h_p', 'h_p', 'e_rel_p'))


		return g



	def getCentroids(self, points):
		N, _ = points.shape
		idx = torch.randperm(N)
		idx_centroids = idx[:self.n_centroids]
		idx_members = idx[self.n_centroids:]
		return points[idx_centroids], points[idx_members]


	def _faissKNNEdge(self, k, index, query=None):
		index = index.numpy().copy(order='C')

		if query is None:
			query = index
		else:
			query = query.numpy()
		
		N = query.shape[0]

		faiss_index = faiss.IndexFlatL2(index.shape[1])
		faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
		faiss_index.add(index)

		if query is None:
			_, k_idx = faiss_index.search(query, k+1)
			k_idx = k_idx[:, 1:]

		else:
			_, k_idx = faiss_index.search(query, k)


		neighbor_idx = np.squeeze(k_idx.reshape(1,-1)).astype(np.int64)
		node_idx = np.concatenate([(np.ones(k) * (i+1) - 1).astype(np.int64) for i in range(N)], axis=0)

		return torch.tensor(node_idx), torch.tensor(neighbor_idx)

		# end_time = time.time()
		# print("Time (sec) to conduct KNN per pc: ", end_time - start_time)

		

	def _get_preprocessed_path(self, filename):
		filename_split= filename.split("/")
		root = "/".join(filename_split[:-4]) + "/graph"

		filename_split[-1] = filename_split[-1][:-4]
		filename = filename_split[-4:]
		filename = "/".join(filename_split[-4:]) + ".pickle"
		filename = "/".join([root, filename])

		return filename

def dst_sub_src(src_field, dst_field, out_field):
	def func(edges):
		return{out_field: edges.dst[dst_field] - edges.src[src_field]}
	return func