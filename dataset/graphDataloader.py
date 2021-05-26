import torch

from dataset.pcDataset import SKPointcloudDataset
from dataset.graphDataset import SKGraphDataset
from configs.config import load_config_data
from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data import DataLoader


def SKGraphDataLoader(args):
	dataset_config = load_config_data(args.dataset_config)
	
	train_pcdataset = SKPointcloudDataset(dataset_config=dataset_config, 
													 split_type='train',
													 label_mapping=args.labelmap_config)
	
	train_graphdataset = SKGraphDataset(pc_dataset=train_pcdataset, 
										dataset_config=dataset_config)
	
	train_dataloader = GraphDataLoader(dataset=train_graphdataset, 
									   batch_size=args.batch_size, 
									   shuffle=False,
									   num_workers=args.num_workers,
									   collate_fn=train_graphdataset.collate_fn)

	return train_dataloader
