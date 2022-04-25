import os
import time
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from matplotlib import pyplot as plt
from utils.init import weight_init
from utils.utils import Accumulator, Timer

from utils.visualize import Animator


lambda_coord = 5.
lambda_noobj = .5


__all__ = ['YoloBackbone', 'Yolo', 'YoloPretrain', 'yolo_loss', 'pretrain', 'train', 'nms']


class YoloBackbone(nn.Module):
	def __init__(self):
		super(YoloBackbone, self).__init__()
		conv1 = nn.Sequential(
			# [#, 448, 448, 3] => [#, 224, 224, 64]
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 224, 224, 64] => [#, 112, 112, 64]
		pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv2 = nn.Sequential(
			# [#, 112, 112, 64] => [#, 112, 112, 192]
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 112, 112, 192] => [#, 56, 56, 192]
		pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv3 = nn.Sequential(
			# [#, 56, 56, 192] => [#, 56, 56, 128]
			nn.Conv2d(192, 128, kernel_size=1),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 56, 56, 128] => [#, 56, 56, 256]
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 56, 56, 256] => [#, 56, 56, 256]
			nn.Conv2d(256, 256, kernel_size=1),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 56, 56, 256] => [#, 56, 56, 512]
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 56, 56, 512] => [#, 28, 28, 512]
		pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		conv4_part = nn.Sequential(
			# [#, 28, 28, 512] => [#, 28, 28, 256]
			nn.Conv2d(512, 256, kernel_size=1),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 28, 28, 256] => [#, 28, 28, 512]
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, inplace=True)
		)
		conv4_modules = []
		for _ in range(4):
			conv4_modules.append(conv4_part)
		conv4 = nn.Sequential(
			*conv4_modules,
			# [#, 28, 28, 512] => [#, 28, 28, 512]
			nn.Conv2d(512, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 28, 28, 512] => [#, 28, 28, 1024]
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 28, 28, 1024] => [#, 14, 14, 1024]
		pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		# [#, 14, 14, 1024] => [#, 14, 14, 1024]
		conv5 = nn.Sequential(
			# [#, 14, 14, 1024] => [#, 14, 14, 512]
			nn.Conv2d(1024, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 512] => [#, 14, 14, 1024]
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 1024] => [#, 14, 14, 512]
			nn.Conv2d(1024, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 512] => [#, 14, 14, 1024]
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True)
		)
		self.net = nn.Sequential(conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5)
	
	def forward(self, X):
		return self.net(X)


class Yolo(nn.Module):
	def __init__(self, backbone: YoloBackbone, backbone_out_channels=1024):
		super(Yolo, self).__init__()
		self.backbone = backbone
		self.head = nn.Sequential(
			# [#, 14, 14, ?] => [#, 14, 14, 1024]
			nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 1024] => [#, 7, 7, 1024]
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 7, 7, 1024] => [#, 7, 7, 1024]
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 7, 7, 1024] => [#, 7, 7, 1024]
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 7, 7, 1024] => [#, 7*7*1024]
			nn.Flatten(),
			# [#, 7*7*1024] => [#, 4096]
			nn.Linear(7*7*1024, 4096),
			# nn.Dropout(0.5),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 4096] => [#, 7*7*30]
			nn.Linear(4096, 7*7*30),
			nn.Sigmoid(), #  normalize to [0, 1]
			# [#, 7*7*30] => [#, 7, 7, 30]
			nn.Unflatten(1, (7, 7, 30))
		)
		self.net = nn.Sequential(self.backbone, self.head)

	def forward(self, X):
		return self.net(X)


class YoloPretrain(nn.Module):
	def __init__(self, backbone: YoloBackbone):
		super(YoloPretrain, self).__init__()
		self.backbone = backbone
		self.head = nn.Sequential(
			# We use 224*224*3 to pretrain on ImageNet
			# so the output is [#, 7, 7, 1024]
			backbone,
			# [#, 7, 7, 1024] => [#, 1, 1, 1024]
			nn.AdaptiveAvgPool2d((1, 1)),
			# [#, 1, 1, 1024] => [#, 1024]
			nn.Flatten(),
			nn.Linear(1024, 1000)
		)
		self.net = nn.Sequential(self.backbone, self.head)

	def forward(self, X):
		return self.net(X)


def yolo_loss(yhat, y):
	"""
	Args:
		yhat: [#, 7, 7, 30]
		y: [#, 7, 7, 30]
	Returns:
		loss: [#]
	"""
	with torch.no_grad():
		# arrange cell xidx, yidx
		# [7, 7]
		cell_xidx = (torch.arange(49) % 7).reshape(7, 7)
		cell_yidx = (torch.div(torch.arange(49), 7, rounding_mode='floor')).reshape(7, 7)
		# transform to [7, 7, 2]
		cell_xidx.unsqueeze_(-1)
		cell_yidx.unsqueeze_(-1)
		cell_xidx.expand(7, 7, 2)
		cell_yidx.expand(7, 7, 2)
		# move to device
		cell_xidx = cell_xidx.to(yhat.device)
		cell_yidx = cell_yidx.to(yhat.device)

	def calc_coord(val):
		with torch.no_grad():
			# transform cell relative coordinates to image relative coordinates
			x = (val[..., 0] + cell_xidx) / 7.0
			y = (val[..., 1] + cell_yidx) / 7.0

			return (x - val[..., 2] / 2.0,
				x + val[..., 2] / 2.0,
				y - val[..., 3] / 2.0,
				y + val[..., 3] / 2.0)

	y_area = y[..., :10].reshape(-1, 7, 7, 2, 5)
	yhat_area = yhat[..., :10].reshape(-1, 7, 7, 2, 5)

	y_class = y[..., 10:].reshape(-1, 7, 7, 20)
	yhat_class = yhat[..., 10:].reshape(-1, 7, 7, 20)

	with torch.no_grad():
		# calculate IoU
		x_min, x_max, y_min, y_max = calc_coord(y_area)
		x_min_hat, x_max_hat, y_min_hat, y_max_hat = calc_coord(yhat_area)

		wi = torch.min(x_max, x_max_hat) - torch.max(x_min, x_min_hat)
		wi = torch.max(wi, torch.zeros_like(wi))
		hi = torch.min(y_max, y_max_hat) - torch.max(y_min, y_min_hat)
		hi = torch.max(hi, torch.zeros_like(hi))

		intersection = wi * hi
		union = (x_max - x_min) * (y_max - y_min) + (x_max_hat - x_min_hat) * (y_max_hat - y_min_hat) - intersection
		iou = intersection / (union + 1e-6) # add epsilon to avoid nan
		
		_, res = iou.max(dim=3, keepdim=True)
	
	# [#, 7, 7, 5]
	# responsible bounding box (having higher IoU)
	yhat_res = torch.take_along_dim(yhat_area, res.unsqueeze(3), 3).squeeze_(3)
	y_res = y_area[..., 0, :5]

	with torch.no_grad():
		# calculate indicator matrix
		have_obj = y_res[..., 4] > 0
		no_obj = ~have_obj
	
	return ((lambda_coord * ( # coordinate loss
		  (y_res[..., 0] - yhat_res[..., 0]) ** 2 # X
		+ (y_res[..., 1] - yhat_res[..., 1]) ** 2 # Y
		+ (torch.sqrt(y_res[..., 2]) - torch.sqrt(yhat_res[..., 2])) ** 2  # W
		+ (torch.sqrt(y_res[..., 3]) - torch.sqrt(yhat_res[..., 3])) ** 2) # H
		# confidence
		+ (y_res[..., 4] - yhat_res[..., 4]) ** 2
		# class
		+ ((y_class - yhat_class) ** 2).sum(dim=3)) * have_obj
		# noobj
		+ ((y_area[..., 0, 4] - yhat_area[..., 0, 4]) ** 2 + \
		(y_area[..., 1, 4] - yhat_area[..., 1, 4]) ** 2) * no_obj * lambda_noobj).sum(dim=(1, 2))


def pretrain(net, train_iter, test_iter, num_epochs, lr, momentum, weight_decay, device):
	# init params
	net.apply(weight_init)
	# copy to device
	net.to(device)
	# define optimizer
	optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	loss = nn.CrossEntropyLoss()
	# train
	# TODO: IMPLEMENT HERE


def train(net, train_iter, test_iter, num_epochs, lr, momentum, weight_decay, device, accum_batch_num=1, save_path='./model', load=None, load_epoch=-1, pretrained=False):
	'''
	Train net work. Some notes for load & load_epoch:
	:param load: the file of model weights to load
	:param load_epoch: num of epoch already completed (minus 1). should be the same with the number in auto-saved file name.
	'''
	
	def print_and_log(msg, log_file):
		print(msg)
		with open(log_file, 'a', encoding='utf8') as f:
			f.write(msg + '\n')

	def update_lr(opt, lr):
		for param_group in opt.param_groups:
			param_group['lr'] = lr

	os.makedirs(save_path, exist_ok=True)
	log_file = os.path.join(save_path, f'log-{time.time_ns()}.txt')

	if load:
		net.load_state_dict(torch.load(load))
	elif pretrained:
		net.head.apply(weight_init)
	else:
		# init params
		net.apply(weight_init)
	
	# copy to device
	net.to(device)
	# define optimizer
	if isinstance(lr, float):
		tlr = lr
	else: tlr = 0.001
	
	optimizer = torch.optim.SGD(net.parameters(), lr=tlr, momentum=momentum, weight_decay=weight_decay)

	# visualization
	animator = Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss', 'test loss'])

	num_batches = len(train_iter)
	# train
	for epoch in range(num_epochs - load_epoch - 1):
		# adjust true epoch number according to pre_load
		epoch = epoch + load_epoch + 1
		
		# define metrics: train loss, sample count
		metrics = Accumulator(2)
		# define timer
		timer = Timer()
		
		# train
		net.train()

		# set batch accumulator
		accum_cnt = 0
		accum = 0
		
		for i, batch in enumerate(train_iter):
			timer.start()

			X, y = batch
			X, y = X.to(device), y.to(device)
			yhat = net(X)

			loss_val = yolo_loss(yhat, y)

			# backward to accumulate gradients
			loss_val.sum().backward()
			# update batch accumulator
			accum += 1
			accum_cnt += loss_val.shape[0]
			# step when accumulator is full
			if accum == accum_batch_num or i == num_batches - 1:
				# update learning rate per epoch and adjust by accumulated batch_size
				if callable(lr):
					update_lr(optimizer, lr(epoch) / accum_cnt)
				else:
					update_lr(optimizer, lr / accum_cnt)
				# step
				optimizer.step()
				# clear
				optimizer.zero_grad()
				accum_cnt = 0
				accum = 0

			# update metrics
			with torch.no_grad():
				metrics.add(loss_val.sum().cpu(), X.shape[0])
			train_l = metrics[0] / metrics[1]
			
			timer.stop()
			
			# log & visualization
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				print_and_log("epoch: %d, batch: %d / %d, loss: %.4f, time: %.4f" % (epoch, i + 1, num_batches, train_l.item(), timer.sum()), log_file)
				animator.add(epoch + (i + 1) / num_batches, (train_l, None))
		
		# redefine metrics: test loss, test sample count
		metrics = Accumulator(2)
		# redefine timer
		timer = Timer()
		# test
		net.eval()

		with torch.no_grad():
			timer.start()
			
			for batch in test_iter:
				X, y = batch
				X, y = X.to(device), y.to(device)
				yhat = net(X)

				loss_val = yolo_loss(yhat, y)
				metrics.add(loss_val.sum().cpu(), X.shape[0])
			
			timer.stop()
			
			test_l = metrics[0] / metrics[1]
			print_and_log("epoch: %d, test loss: %.4f, time: %.4f" % (epoch + 1, test_l.item(), timer.sum()), log_file)
			animator.add(epoch + 1, (None, test_l))

		# save model
		torch.save(net.state_dict(), os.path.join(save_path, f'./{time.time_ns()}-epoch-{epoch}.pth'))


def nms(pred, threshold=0.5):
	'''
	Non-maximum suppression directly for output.
	:param pred: pred results
	:param threshold:
	:return:
	'''
	with torch.no_grad():
		pred = pred.reshape((-1, 30))
		# [[idx, x, y, w, h, iou, score_cls]]
		nms_data = [[] for _ in range(20)]
		for i in range(pred.shape[0]):
			cell = pred[i]
			score, idx = torch.max(cell[10:30], dim=0)
			idx = idx.item()
			x, y, w, h, iou = cell[0:5].cpu().numpy()

			nms_data[idx].append([i, x, y, w, h, iou, score.item()])
			x, y, w, h, iou = cell[5:10].cpu().numpy()
			nms_data[idx].append([i, x, y, w, h, iou, score.item()])

		ret = torch.zeros_like(pred)
		flag = torch.zeros(pred.shape[0], dtype=torch.bool)
		for c in range(20):
			c_nms_data = np.array(nms_data[c])

			keep_index = _nms(c_nms_data, threshold)
			keeps = c_nms_data[keep_index]

			for keep in keeps:
				i, x, y, w, h, iou, score = keep
				i = int(i)
				
				last_score, _ = torch.max(ret[i][10:30], dim=0)
				last_iou = ret[i][4]

				if score * iou > last_score * last_iou:
					flag[i] = False
				if flag[i]: continue
				
				ret[i][0:5] = torch.tensor([x, y, w, h, iou])
				ret[i][10:30] = 0
				ret[i][10 + c] = score

				flag[i] = True

		return ret


def _nms(data, threshold):
	'''
	Non-maximum suppression.
	:param data: numpy data array (i, x, y, w, h, score_area, score_cls)
	:param threshold:
	:return: keep index array
	'''
	if len(data) == 0:
		return []

	# cell relative coordinates
	cell_idx = data[:, 0]
	x = data[:, 1]
	y = data[:, 2]
	# calculate cell index
	xidx = cell_idx % 7
	yidx = cell_idx // 7
	# transform to image relative coordinates
	x = (x + xidx) / 7.0
	y = (y + yidx) / 7.0
	# obtain image relative width & height
	w = data[:, 3]
	h = data[:, 4]
	# calculate coordinates
	x1 = x - w / 2
	y1 = y - h / 2
	x2 = x + w / 2
	y2 = y + h / 2

	score_area = data[:, 5]

	areas = w * h
	
	order = score_area.argsort()[::-1]
	keep = []

	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1)
		h = np.maximum(0.0, yy2 - yy1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)

		inds = np.where(ovr <= threshold)[0]
		order = order[inds + 1]
	
	return keep
