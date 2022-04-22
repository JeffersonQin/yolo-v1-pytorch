import json
import torch
import torchvision
import random
import cv2
import numpy
from PIL import Image
from matplotlib import pyplot as plt
from IPython import display


__all__ = ['draw_box', 'draw_detection_result', 'draw_ground_truth', 'PIL_to_cv2', 'cv2_to_PIL', 'tensor_to_PIL', 'tensor_to_cv2', 'Animator']


categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# 20 random color for labeling
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]


def draw_box(img, x, y, w, h, score, category):
	"""
	Tool function to draw confidence box on image
	:param img: numpy image to be rendered
	:param x, y: relative center of box
	:param w, h: relative size of box
	:param score: confidence score
	:param category: category of object
	:return: image with box
	"""
	height = img.shape[0] * h
	width = img.shape[1] * w
	left = img.shape[1] * x - width / 2
	top = img.shape[0] * y - height / 2

	color = colors[category]
	text = categories[category] + " " + str(float(score))
	cv2.rectangle(img, (int(left), int(top)), (int(left + width), int(top + height)), color, 2)

	text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
	p1 = (int(left), int(top) - text_size[1])

	cv2.rectangle(img,
		(p1[0] - 2//2, p1[1] - 2 - baseline),
		(p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
	cv2.putText(img, text,
		(p1[0], p1[1] + baseline),
		cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
	
	return img


def draw_detection_result(img, pred, raw=False, thres=0.1):
	"""
	Tool function to draw detection result on image
	:param img: numpy image to be rendered
	:param pred: detection result (torch.Tensor)
	:param raw: if true, two dimension of detection results will be rendered (5 * 2 + 20)
	:param thres: threshold to filter out low confidence boxes
	:return: image with detection result
	"""
	if raw:
		offsets = [0, 5]
	else: offsets = [0]

	for offset in offsets:
		pred = pred.reshape((-1, 30))
		for i in range(pred.shape[0]):
			x, y, w, h, iou = pred[i][0 + offset : 5 + offset]

			# calculate cell index
			xidx = i % 7
			yidx = i // 7

			# transform cell relative coordinates to image relative coordinates
			x = (x + xidx) / 7.0
			y = (y + yidx) / 7.0

			score, cat = pred[i][10:30].max(dim=0)
			if iou * score < thres: continue
			img = draw_box(img, x, y, w, h, score * iou, cat)

	return img


def draw_ground_truth(img, truth):
	"""
	Tool function to draw ground truth
	:param img: numpy image to be rendered
	:param pred: truth bbox in json format (str)
	:return: image with ground truth bbox
	"""
	pred = json.loads(truth)
	for bbox in pred:
		xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
		w = xmax - xmin
		h = ymax - ymin
		x = (xmin + xmax) / 2
		y = (ymin + ymax) / 2
		img = draw_box(img, x, y, w, h, 1, bbox['category'])
	return img


def tensor_to_PIL(img):
	"""Convert a tensor into a PIL image"""
	to_pil = torchvision.transforms.ToPILImage()
	return to_pil(img.cpu()).convert('RGB')


def tensor_to_cv2(img):
	return PIL_to_cv2(tensor_to_PIL(img))


def PIL_to_cv2(img):
	"""
	Tool function to convert PIL image to cv2 image
	:param img: PIL image
	:return: cv2 image
	"""
	img = numpy.array(img)
	img = img[:, :, ::-1].copy()
	return img


def cv2_to_PIL(img):
	"""
	Tool function to convert cv2 image to PIL image
	:param img: cv2 image
	:return: PIL image
	"""
	img = img[:, :, ::-1].copy()
	img = Image.fromarray(img)
	return img


# from d2l
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()


# from d2l
class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
