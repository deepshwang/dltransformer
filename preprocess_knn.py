import torch
from dataset.ModelNet40 import ModelNet40Dataset
import pdb
import numpy as np
import faiss
import time
import pgt_ops.pgt_utils as pt_utils
import msgpack
import msgpack_numpy
from utils.tuil import open_yaml


def main(mode):
	train_dataset = ModelNet40Dataset(train=(mode=='train'))
	# k-means ratio
	k = 16

	# downsampling ratio
	d = 4


	# expansion ratio
	e = 2

	root_dir = '/media/TrainDataset/modelnet40_normal_resampled_cache/index_files/k_' + str(k) + "_d_" + str(d) + "_e_" + str(e) 
	
	for i in range(len(train_dataset)):
		point, _, cluster_idx, ds_idx = train_dataset[i]
		pos = point[:, :3]
		k_idx_list = []
		for l in range(4):
			pos_cuda = torch.unsqueeze(torch.from_numpy(pos), 0).to('cuda:0')
			ds_idx, pos_ds = FPS(pos_cuda, d)
			pos_ds = np.squeeze(pos_ds.cpu().numpy())
			k_idx = faissKNN(index=pos, query=pos_ds, k=k)
			k_idx_list.append(k_idx)
			pos = pos_ds

		num = str(i)
		while len(num) < 4:
			num = '0' + num
		fps_knn_filename = root_dir + '/' + mode + '/fps_knn_' + num + '.msgpack'

		with open(fps_knn_filename, "wb") as outfile:
			packed = msgpack.packb(k_idx_list, default=msgpack_numpy.encode)
			outfile.write(packed)

		if i % 5 == 0:
			print(i+1, ' / ', len(train_dataset))





def FPS(pos, downsample_ratio):
	B, N, _ = pos.shape
	fp_idx = pt_utils.farthest_point_sample(pos.contiguous(), int(N/downsample_ratio))
	pos_downsampled = gather_by_idx(pos, fp_idx)
	return fp_idx, pos_downsampled



def faissKNN(index, query, k):
	index = index.copy(order='C')

	# query = query.cpu().numpy()
	
	N = query.shape[0]

	faiss_index = faiss.IndexFlatL2(index.shape[1])
	faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
	faiss_index.add(index)
	_, k_idx = faiss_index.search(query, k)


	# neighbor_idx = np.squeeze(k_idx.reshape(1,-1)).astype(np.int64)
	# node_idx = np.concatenate([(np.ones(k) * (i+1) - 1).astype(np.int64) for i in range(N)], axis=0)
	return k_idx


def gather_by_idx(db, q_idx):
	db_flipped = torch.einsum("ijk->ikj", db).contiguous()
	db = pt_utils.gather_operation(db_flipped, q_idx)
	db = torch.einsum("ijk->ikj", db).contiguous()
	return db



if __name__ == '__main__':
	main('train')
	main('test')