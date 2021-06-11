import torch
from dataset.ModelNet40 import ModelNet40Dataset
import pdb
import numpy as np
import faiss
import time
import pgt_ops.pgt_utils as pt_utils
import msgpack
import msgpack_numpy


def FPS(pos, downsample_ratio):
	B, N, _ = pos.shape
	fp_idx = pt_utils.farthest_point_sample(pos.contiguous(), int(N/downsample_ratio))
	pos_downsampled = gather_by_idx(pos, fp_idx)
	return fp_idx, pos_downsampled


def gather_by_idx(db, q_idx):
	db_flipped = torch.einsum("ijk->ikj", db).contiguous()
	db = pt_utils.gather_operation(db_flipped, q_idx)
	db = torch.einsum("ijk->ikj", db).contiguous()
	return db

def faissKNN(index, query, k):
	index = np.squeeze(index.cpu().numpy().copy(order='C'))

	# query = query.cpu().numpy()
	
	N = query.shape[0]

	faiss_index = faiss.IndexFlatL2(index.shape[1])
	faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
	faiss_index.add(index)
	_, k_idx = faiss_index.search(query, k)


	# neighbor_idx = np.squeeze(k_idx.reshape(1,-1)).astype(np.int64)
	# node_idx = np.concatenate([(np.ones(k) * (i+1) - 1).astype(np.int64) for i in range(N)], axis=0)
	return k_idx


def faissKMeans(pos, kmeans_ratio):
	N, C = pos.shape
	pos = pos.astype(np.float32)
	faiss_kmeans = faiss.Kmeans(d=C, k=int(N/kmeans_ratio), niter=150, verbose=False, gpu=True)
	faiss_kmeans.min_points_per_centroid=1
	faiss_kmeans.train(pos)
	dists, idxs = faiss_kmeans.index.search(pos, 1)
	return idxs



def get_cluster_idxes(pos, kmeans_ratio):
	N, C = pos.shape

	cluster_assignment_idxs = faissKMeans(pos, kmeans_ratio)
	cluster_idx_dict={}
	for i, cluster_idx in enumerate(np.squeeze(cluster_assignment_idxs)):
		cluster_idx = int(cluster_idx)
		if cluster_idx not in cluster_idx_dict.keys():
			cluster_idx_dict[cluster_idx] = [i]
		else:
			cluster_idx_dict[cluster_idx].append(i)

	return cluster_idx_dict



def main(mode):
	# k-means ratio
	k = 16

	# downsampling ratio
	d = 4


	# expansion ratio
	e = 4
	train_dataset = ModelNet40Dataset(train=(mode=='train'), preprocess=False, k=k, d=d, e=e)


	root_dir = '/media/TrainDataset/modelnet40_normal_resampled_cache/index_files/k_' + str(k) + "_d_" + str(d) + "_e_" + str(e) 
	
	for i in range(len(train_dataset)):
		point, _ = train_dataset[i]
		pos = point[:,:3]
		instance_cluster_list = []
		ds_idx_initial = []
		k_idx_list = []
		for l in range(4):
			# [1] Store cluster index as a list of dictionaries
			cluster_idx_dict_prev = get_cluster_idxes(pos, k)
			instance_cluster_list.append(cluster_idx_dict_prev)

			cluster_idx_dict = get_cluster_idxes(pos, e*k)
			instance_cluster_list.append(cluster_idx_dict)

			pos = torch.unsqueeze(torch.from_numpy(pos), 0).to('cuda:0')
			ds_idx, pos_ds = FPS(pos, d)

			pos_ds = np.squeeze(pos_ds.cpu().numpy())
			k_idx = faissKNN(index=pos, query=pos_ds, k=k)
			k_idx_list.append(k_idx)
			pos = torch.from_numpy(pos_ds).to('cuda:0')

			pos = torch.squeeze(pos).cpu().numpy()
			if l==0:
				ds_idx_initial = ds_idx.cpu().numpy()

		num = str(i)
		while len(num) < 4:
			num = '0' + num
		cluster_filename = root_dir + '/' + mode + '/cluster_' + num + '.msgpack'
		downsample_filename = root_dir + '/' + mode + '/downsample_' + num + '.msgpack'
		fps_knn_filename = root_dir + '/' + mode + '/fps_knn_' + num + '.msgpack'


		with open(cluster_filename, "wb") as outfile:
			packed = msgpack.packb(instance_cluster_list)
			outfile.write(packed)

		with open(downsample_filename, "wb") as outfile:
			packed = msgpack.packb(ds_idx_initial, default=msgpack_numpy.encode)
			outfile.write(packed)

		with open(fps_knn_filename, "wb") as outfile:
			packed = msgpack.packb(k_idx_list, default=msgpack_numpy.encode)
			outfile.write(packed)


		if i % 5 == 0:
			print(i+1, ' / ', len(train_dataset))







if __name__ == '__main__':
	main('train')
	main('test')