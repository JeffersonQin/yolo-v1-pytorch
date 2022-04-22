import math
import torch
import torchvision
import random
from torch.utils import data


__all__ = ['VOCDataset', 'load_data_voc']


categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataset(data.Dataset):
	def __init__(self, dataset, train=True):
		self.dataset = dataset
		self.train = train
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		img, target = self.dataset[idx]
		
		if not isinstance(target['annotation']['object'], list):
			target['annotation']['object'] = [target['annotation']['object']]
		count = len(target['annotation']['object'])

		height, width = int(target['annotation']['size']['height']), int(target['annotation']['size']['width'])

		# Image Augmentation
		if self.train:
			# randomly scaling and translation up to 20%
			if random.random() < 0.5:
				# use random value to decide scaling factor on x and y axis
				random_height = random.random() * 0.2
				random_width = random.random() * 0.2
				# use random value again to decide scaling factor for 4 borders
				random_top = random.random() * random_height
				random_left = random.random() * random_width
				# calculate new width and height and position
				top = random_top * height
				left = random_left * width
				height = height - random_height * height
				width = width - random_width * width
				# crop image
				img = torchvision.transforms.functional.crop(img, int(top), int(left), int(height), int(width))
			
				# update target
				for i in range(count):
					obj = target['annotation']['object'][i]
					obj['bndbox']['xmin'] = max(0, float(obj['bndbox']['xmin']) - left)
					obj['bndbox']['ymin'] = max(0, float(obj['bndbox']['ymin']) - top)
					obj['bndbox']['xmax'] = min(width, float(obj['bndbox']['xmax']) - left)
					obj['bndbox']['ymax'] = min(height, float(obj['bndbox']['ymax']) - top)
			
			# adjust saturation randomly up to 150%
			if random.random() < 0.5:
				random_saturation = random.random() + 0.5
				img = torchvision.transforms.functional.adjust_saturation(img, random_saturation)
		
		# resize to 448*448
		img = torchvision.transforms.functional.resize(img, (448, 448))

		# update labels from absolute to relative
		height, width = float(height), float(width)

		for i in range(count):
			obj = target['annotation']['object'][i]
			obj['bndbox']['xmin'] = float(obj['bndbox']['xmin']) / width
			obj['bndbox']['ymin'] = float(obj['bndbox']['ymin']) / height
			obj['bndbox']['xmax'] = float(obj['bndbox']['xmax']) / width
			obj['bndbox']['ymax'] = float(obj['bndbox']['ymax']) / height

		# Label Encoding
		# [{'name': '', 'xmin': '', 'ymin': '', 'xmax': '', 'ymax': '', }, {}, {}, ...]
		# ==>
		# [x, y  (relative to cell), width, height, 1 if exist (confidence),
		#  x, y  (relative to cell), width, height, 1 if exist (confidence),
		#  one-hot encoding of 20 categories]
		label = torch.zeros((7, 7, 30))
		for i in range(count):
			obj = target['annotation']['object'][i]
			xmin = obj['bndbox']['xmin']
			ymin = obj['bndbox']['ymin']
			xmax = obj['bndbox']['xmax']
			ymax = obj['bndbox']['ymax']
			name = obj['name']

			if xmin == xmax or ymin == ymax:
				continue
			if xmin >= 1 or ymin >= 1 or xmax <= 0 or ymax <= 0:
				continue
			
			x = (xmin + xmax) / 2.0
			y = (ymin + ymax) / 2.0

			width = xmax - xmin
			height = ymax - ymin

			xidx = math.floor(x * 7.0)
			yidx = math.floor(y * 7.0)
			

			# According to the paper
			# if multiple objects exist in the same cell
			# pick the one with the largest area
			if label[yidx][xidx][4] == 1: # already have object
				if label[yidx][xidx][2] * label[yidx][xidx][3] < width * height:
					use_data = True
				else: use_data = False
			else: use_data = True

			if use_data:
				for offset in [0, 5]:
					# Transforming image relative coordinates to cell relative coordinates:
					# x - idx / 7.0 = x_cell / cell_count (7.0)
					# => x_cell = x * cell_count - idx = x * 7.0 - idx
					# y is the same
					label[yidx][xidx][0 + offset] = x * 7.0 - xidx
					label[yidx][xidx][1 + offset] = y * 7.0 - yidx
					label[yidx][xidx][2 + offset] = width
					label[yidx][xidx][3 + offset] = height
					label[yidx][xidx][4 + offset] = 1
				label[yidx][xidx][10 + categories.index(name)] = 1

		return img, label


def load_data_voc(batch_size, num_workers=0, persistent_workers=False, download=False):
	"""
	Loads the Pascal VOC dataset.
	"""
	# Load the dataset
	trans = [
		torchvision.transforms.ToTensor(),
	]
	torchvision.transforms
	trans = torchvision.transforms.Compose(trans)
	voc2007_trainval = torchvision.datasets.VOCDetection(root='./data/VOCDetection/', year='2007', image_set='trainval', download=download, transform=trans)
	voc2007_test = torchvision.datasets.VOCDetection(root='./data/VOCDetection/', year='2007', image_set='test', download=download, transform=trans)
	voc2012_train = torchvision.datasets.VOCDetection(root='./data/VOCDetection/', year='2012', image_set='train', download=download, transform=trans)
	voc2012_val = torchvision.datasets.VOCDetection(root='./data/VOCDetection/', year='2012', image_set='val', download=download, transform=trans)
	return (
		data.DataLoader(VOCDataset(data.ConcatDataset([voc2007_trainval, voc2007_test, voc2012_train]), train=True), 
			batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers), 
		data.DataLoader(VOCDataset(voc2012_val, train=False), 
			batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
	)
