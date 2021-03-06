import argparse
import torch
from datetime import datetime
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
						default=1e-1)

	parser.add_argument('--gamma', 
						type=float,
						default=0.5)

	parser.add_argument('--DLPT_config',
						type=str,
						default='./configs/DLPT_modelconfig.yaml')

	parser.add_argument('--model_config_type',
						type=str,
						default='b')


	parser.add_argument('--load_checkpoint',
						type=str,
						default=None)

	# parser.add_argument('--plot_name',
	# 					type=str,
	# 					default='Loss Plot')


	parser.add_argument('--pre_ln',
						type=bool,
						default=False)


	parser.add_argument('--shuffle',
						type=bool,
						default=True,
						help='Whether to shuffle training dataloader')


	parser.add_argument('--sanity_check',
						type=bool,
						default=False,
						help='Whether to train a model to overfit a single batch to see if the model has the capability to learn')

	args = parser.parse_args()


	now = datetime.now()
	dt_string = now.strftime("%d-%m-%Y_%H:%M")

	
	parser.add_argument('--plot_name',
					  type=str,
					  default=dt_string + "_gamma_" + str(args.gamma) + "_lr_" + str(args.lr)  + "_modelconfig_" + args.model_config_type)


	parser.add_argument('--model_save_name',
					type=str,
					default= './state_dict/' + dt_string + "_gamma_" + str(args.gamma) + "_lr_" + str(args.lr) + "_modelconfig_" + args.model_config_type + ".pt")





	args = parser.parse_args()

	return args