import torch
import numpy as np

from models.modules import DLPTNet_cls
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
	def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer, epochs, args):
		self.model = model
		self.train_dataloader = train_dataloader
		self.test_dataloader = test_dataloader
		self.criterion = criterion
		self.optimizer = optimizer
		self.epochs = epochs
		self.args = args

	def train(self):
		self.vis = visdom.Visdom()
		loss_plt = self.vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
		best_train_acc = 0.0
		self.model = self.model.to(self.args.device)

		for e in range(self.epochs):
			# Train
			print("Resume Training: ", str(e + 1) , " / ", str(self.epochs))
			self.model.train()
			running_loss = 0.0
			for i, (points, labels, cluster_idx, ds_idx) in enumerate(self.train_dataloader):
				points = points.to(self.args.device)
				labels = labels.to(self.args.device)
				ds_idx = [d.to(self.args.device) for d in ds_idx] 

				outputs = self.model(points, cluster_idx, ds_idx)
				
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

			# Test
			total = 0.0
			correct = 0.0
			self.model.eval()
			with torch.no_grad():
				for i, (points, labels, cluster_idx, ds_idx) in enumerate(self.test_dataloader):
					if i == 0:
						points = points.to(self.args.device)
						labels = labels.to(self.args.device)
						ds_idx = [d.to(self.args.device) for d in ds_idx] 

						outputs = self.model(points, cluster_idx, ds_idx)
						loss = self.criterion(outputs, labels)
						_, preds = torch.max(outputs, dim=1)
						total += labels.size(0)
						correct += (preds == labels).sum().item()


				train_acc = (100 * correct / total)
				print("[", e + 1, "/ ", self.epochs, "]  Acc: ", train_acc, "%")


				if train_acc > best_train_acc:
					torch.save(self.model.state_dict, self.args.model_save_name)
					best_train_acc = train_acc
					print("MODEL UPGRADED!")









	def value_tracker(self, value_plot, value, num):
		'''num, loss_value, are Tensor'''
		self.vis.line(X=num,
					Y=value,
					win = value_plot,
					update='append'
					)


def main(args):


	train_dataloader = ModelNet40DataLoader(args, 
											num_points=1024,
											shuffle=True,
											train=True,
											transforms=T_train)
	test_dataloader = ModelNet40DataLoader(args,
										   num_points=1024,
										   shuffle=True,
										   train=False,
										   transforms=T_test)

	model = DLPTNet_cls(open_yaml(args.DLPT_config)['layer_params']['a'], c=40)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	criterion = torch.nn.CrossEntropyLoss()

	trainer = Trainer(model=model,
					  train_dataloader=train_dataloader,
					  test_dataloader=test_dataloader,
					  criterion=criterion,
					  optimizer=optimizer,
					  epochs=args.epochs,
					  args=args)

	trainer.train()





if __name__ == '__main__':
	args = argument_parser()
	main(args)