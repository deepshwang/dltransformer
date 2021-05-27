import argparse
import torch

def argument_parser():
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('--dataset_config', 
						default='configs/semanticKitti_datasetconfig.yaml')

	parser.add_argument('--model_config', 
						default='configs/semanticKitti_modelconfig.yaml')

	parser.add_argument('--labelmap_config', 
						default='configs/semanticKitti_labelmapconfig.yaml')

	parser.add_argument('--device', 
						default='cuda' if torch.cuda.is_available() else 'cpu')

	parser.add_argument('--train_batch_size', 
						type=int,
						default=16)

	parser.add_argument('--test_batch_size', 
						type=int,
						default=64)

	parser.add_argument('--epochs', 
						type=int,
						default=250)

	parser.add_argument('--num_workers', 
						default=4)

	parser.add_argument('--step_size', 
						type=int,
						default=20)

	parser.add_argument('--lr', 
						type=float,
						default=1e-4)

	parser.add_argument('--gamma', 
						type=float,
						default=0.5)

	parser.add_argument('--DLPT_config',
						type=str,
						default='./configs/DLPT_modelconfig.yaml')

	parser.add_argument('--model_config_type',
						type=str,
						default='b')

	parser.add_argument('--model_save_name',
						type=str,
						default='./state_dict/best_net.pth')

	args = parser.parse_args()

	return args