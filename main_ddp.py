import torch
import numpy as np
import torch.utils.data as data


from models.modules import DLPTNet_cls
from dataset.ModelNet40 import ModelNet40DataLoader, ModelNet40Dataset
from dataset.transforms import T_modelnet_train as T_train
from dataset.transforms import T_modelnet_test as T_test
import utils.data_utils as d_utils
import pdb
import yaml
from torchvision import transforms
from utils.argparse import argument_parser
from utils.util import open_yaml
import visdom
import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



class Trainer():
	def __init__(self, model, train_dataset, test_dataset, criterion, epochs, args):
		self.model = model
		self.ddp_model = None
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.criterion = criterion
		self.epochs = epochs
		self.args = args

	def train(self, rank, world_size):
		print(f"Running basic DDP on rank {rank}.")
		self.setup(rank, world_size)

		self.vis = visdom.Visdom()
		loss_plt = self.vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
		best_train_acc = 0.0
		self.model = self.model.to(rank)
		self.ddp_model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

		self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=self.args.lr)

		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma, verbose=True)


		sampler = DistributedSampler(self.train_dataset)
		self.train_dataloader = self.ModelNet40DataLoaderDDP(args=self.args, 
															 dataset=self.train_dataset,
															 sampler=sampler,
															 train=True)


		for e in range(self.epochs):
			# Train
			print("Resume Training: ", str(e + 1) , " / ", str(self.epochs))
			self.ddp_model.train()
			dist.barrier()

			running_loss = 0.0
			for i, (points, labels, cluster_idx, ds_idx) in enumerate(self.train_dataloader):
				points = points.to(rank)
				labels = labels.to(rank)
				ds_idx = [d.to(rank) for d in ds_idx] 

				outputs = self.ddp_model(points, cluster_idx, ds_idx)
				
				loss = self.criterion(outputs, labels)

				self.value_tracker(loss_plt,
								  np.array([loss.detach().cpu()]),
								  np.array([i/len(self.train_dataloader) + e]))

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()
				if (i+1) % 1000 == 0:    
					print('[%d, %5d / %d] loss: %.3f' %
						(e + 1, i + 1, len(self.train_dataloader) ,running_loss/1000))
					running_loss = 0.0

			self.scheduler.step()

			# Test
			total = 0.0
			correct = 0.0
			self.ddp_model.eval()
			with torch.no_grad():
				for i, (points, labels, cluster_idx, ds_idx) in enumerate(self.test_dataloader):
					points = points.to(rank)
					labels = labels.to(rank)
					ds_idx = [d.to(rank) for d in ds_idx] 

					outputs = self.ddp_model(points, cluster_idx, ds_idx)
					loss = self.criterion(outputs, labels)
					_, preds = torch.max(outputs, dim=1)
					total += labels.size(0)
					correct += (preds == labels).sum().item()


				train_acc = (100 * correct / total)
				print("[", e + 1, "/ ", self.epochs, "]  Acc: ", train_acc, "%")


				if train_acc > best_train_acc:
					self.save_checkpoint(rank)
					best_train_acc = train_acc
					print("MODEL UPGRADED!")

		self.cleanup()


	def save_checkpoint(self, rank):
		if rank == 0:
			torch.save(self.ddp_model.state_dict(), self.args.model_save_name)
		dist.barrier()
		return None



	def value_tracker(self, value_plot, value, num):
		'''num, loss_value, are Tensor'''
		self.vis.line(X=num,
					Y=value,
					win = value_plot,
					update='append'
					)

	def setup(self, rank, world_size):
	    os.environ['MASTER_ADDR'] = 'localhost'
	    os.environ['MASTER_PORT'] = '12355'

	    # initialize the process group
	    dist.init_process_group("gloo", rank=rank, world_size=world_size)

	def cleanup(self):
	    dist.destroy_process_group()

	def ModelNet40DataLoaderDDP(self, args=None, dataset=None, sampler=None, train=None):
	    if train:
	        batch_size = args.train_batch_size
	    else:
	        batch_size = args.test_batch_size


	    dataloader = data.DataLoader(dataset=dataset,
	                    batch_size=batch_size,
	                    sampler=sampler,
	                    num_workers=args.num_workers,
	                    shuffle=(sampler is None),
	                    collate_fn=dataset.collate_fn)
	    return dataloader


def main(args):
	n_gpus = torch.cuda.device_count()
	print("Number of GPU device available: ", n_gpus)

	train_dataset = ModelNet40Dataset(num_points=1024, transforms=T_train, train=True)

	test_dataset = ModelNet40Dataset(num_points=1024, transforms=T_test, train=False)

	model = DLPTNet_cls(open_yaml(args.DLPT_config)['layer_params'][args.model_config_type], c=40)

	criterion = torch.nn.CrossEntropyLoss()

	trainer = Trainer(model=model,
					  train_dataset=train_dataset,
					  test_dataset=test_dataset,
					  criterion=criterion,
					  epochs=args.epochs,
					  args=args)

	world_size = 2
	mp.spawn(trainer.train,
			 args=(world_size,),
			 nprocs=world_size,
			 join=True)






if __name__ == '__main__':
	args = argument_parser()
	main(args)
