import torch
import numpy as np

from models.modules import DLPTNet_cls, DLPTNet_PreLN_cls
from dataset.ModelNet40 import ModelNet40DataLoader
from dataset.transforms import T_modelnet_train as T_train
from dataset.transforms import T_modelnet_test as T_test
import utils.data_utils as d_utils
import pdb
import yaml
from torchvision import transforms
from utils.argparse import argument_parser
from utils.util import open_yaml
import visdom



class Trainer():
	def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, start_epoch, epochs, args):
		self.model = model
		self.train_dataloader = train_dataloader
		self.test_dataloader = test_dataloader
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.start_epoch = start_epoch
		self.epochs = epochs
		self.args = args

	def train(self):
		self.vis = visdom.Visdom()
		loss_plt = self.vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title=self.args.plot_name, legend=['Training Loss'], showlegend=True))
		val_loss_plt = self.vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title="Val_loss_" + self.args.plot_name, legend=['Validation Loss'], showlegend=True))
		val_acc_plt = self.vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title="Val_acc_" + self.args.plot_name, legend=['Validation Accuracy'], showlegend=True))
		best_train_acc = 0.0
		self.model = self.model.to(self.args.device)

		for e in range(self.start_epoch, self.epochs):
			# # Train
			print("Resume Training: ", str(e + 1) , " / ", str(self.epochs))
			self.model.train()
			running_loss = 0.0
			for i, (points, labels, cluster_idx, ds_idx, fpsknn_idx) in enumerate(self.train_dataloader):
				points = points.to(self.args.device)
				labels = labels.to(self.args.device)
				ds_idx = [d.to(self.args.device) for d in ds_idx] 

				outputs = self.model(points, cluster_idx, ds_idx, fpsknn_idx)
				
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
			val_running_loss = 0.0
			self.model.eval()
			with torch.no_grad():
				for i, (points, labels, cluster_idx, ds_idx, fpsknn_idx) in enumerate(self.test_dataloader):
					points = points.to(self.args.device)
					labels = labels.to(self.args.device)
					ds_idx = [d.to(self.args.device) for d in ds_idx] 

					outputs = self.model(points, cluster_idx, ds_idx, fpsknn_idx)
					loss = self.criterion(outputs, labels)
					val_running_loss += loss
					_, preds = torch.max(outputs, dim=1)
					total += labels.size(0)
					correct += (preds == labels).sum().item()


				val_acc = (100 * correct / total)
				print("[", e + 1, "/ ", self.epochs, "]  Acc: ", val_acc, "%")
				
				self.value_tracker(val_acc_plt,
				  np.array([val_acc]),
				  np.array([1 + e]))

				self.value_tracker(val_loss_plt,
								  np.array([val_running_loss.detach().cpu()]),
								  np.array([1 + e]))

				if val_acc > best_train_acc:
					torch.save({'epoch': e,
								'model_state_dict': self.model.state_dict(),
								'optimizer_state_dict': self.optimizer.state_dict(),
								'scheduler_state_dict': self.scheduler.state_dict()
						}, self.args.model_save_name)
					best_train_acc = val_acc
					print("MODEL UPGRADED!")


	def train_single_batch(self):
		best_train_acc = 0.0
		self.model = self.model.to(self.args.device)
		
		# debug
		model_prev = None
		same_block_name = []
		different_block_name = []

		points, labels, cluster_idx, ds_idx, fpsknn_idx = next(iter(self.train_dataloader))

		for e in range(self.epochs):
			# Train
			self.model.train()
			running_loss = 0.0
			points = points.to(self.args.device)
			labels = labels.to(self.args.device)
			ds_idx = [d.to(self.args.device) for d in ds_idx] 

			outputs = self.model(points, cluster_idx, ds_idx, fpsknn_idx)
			
			loss = self.criterion(outputs, labels)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()


			# if model_prev is not None:
			# 	for (name1, p1), (name2, p2) in zip(self.model.named_parameters(), model_prev.named_parameters()):
			# 		if p1.data.ne(p2.data).sum() == 0:
			# 			different_block_name.append(name1)
			# 		else:
			# 			same_block_name.append(name1)
			# 	print("different weight name: ")
			# 	print(different_block_name)
			# 	print("same weight name: ")
			# 	print(same_block_name)


   
			print('[%d / %d] loss: %.3f' %
				(e + 1, self.epochs+1 ,loss.item()))

			self.scheduler.step()
			model_prev = self.model







	def value_tracker(self, value_plot, value, num):
		'''num, loss_value, are Tensor'''
		self.vis.line(X=num,
					Y=value,
					win = value_plot,
					update='append'
					)


def main(args):
	kde_config = open_yaml(args.DLPT_config)['layer_params'][args.model_config_type]
	k = kde_config['k']
	d = kde_config['ds_ratio']
	e = kde_config['expansion_ratio']

	train_dataloader = ModelNet40DataLoader(args, 
											num_points=1024,
											shuffle=args.shuffle,
											train=True,
											transforms=T_train, k=k, d=d, e=e)
	test_dataloader = ModelNet40DataLoader(args,
										   num_points=1024,
										   shuffle=False,
										   train=False,
										   transforms=T_test, k=k, d=d, e=e)
	if args.pre_ln:
		model = DLPTNet_PreLN_cls(open_yaml(args.DLPT_config)['layer_params'][args.model_config_type], c=40)
		print("PreLN Mode")
	else:
		model = DLPTNet_cls(open_yaml(args.DLPT_config)['layer_params'][args.model_config_type], c=40)


	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, verbose=True)

	criterion = torch.nn.CrossEntropyLoss()

	start_epoch = 0

	if args.load_checkpoint is not None:
		print("=== Resume training from loading: ", args.load_checkpoint, " ===")
		checkpoint = torch.load(args.load_checkpoint)
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, verbose=True)
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	trainer = Trainer(model=model,
					  train_dataloader=train_dataloader,
					  test_dataloader=test_dataloader,
					  criterion=criterion,
					  optimizer=optimizer,
					  scheduler=scheduler,
					  start_epoch=start_epoch,
					  epochs=args.epochs,
					  args=args)

	if args.sanity_check:
		trainer.train_single_batch()
	else:
		trainer.train()






if __name__ == '__main__':
	args = argument_parser()
	main(args)
