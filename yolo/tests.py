import cv2
import torch
import torchvision
from torch.utils import data
from .yolo import nms
from IPython import display
from utils.visualize import *
from utils.metrics import *


def test_one_batch(net: torch.nn.Module, test_iter_raw: data.DataLoader):
	"""Display result of one test batch

	Args:
		net (torch.nn.Module): network
		test_iter_raw (data.DataLoader): test dataloader (raw)
	"""
	net.eval()

	for X, YRaw in test_iter_raw:
		YHat = net(X)
		for x, yhat, yraw in zip(X, YHat, YRaw):
			yhat = nms(yhat)
			display.display(cv2_to_PIL(draw_detection_result(tensor_to_cv2(x), yhat, raw=False, thres=0.1)))
			display.display(cv2_to_PIL(draw_ground_truth(tensor_to_cv2(x), yraw)))
		break


def test_and_draw_mAP(net: torch.nn.Module, test_iter_raw: data.DataLoader):
	"""Calculate VOCmAP and COCOmAP on test dataset, and draw VOC-AP for each category

	Args:
		net (torch.nn.Module): network
		test_iter_raw (data.DataLoader): test dataloader (raw)
	"""
	net.eval()

	# metrics calculation
	calc = ObjectDetectionMetricsCalculator(20, 0.1)

	for i, (X, YRaw) in enumerate(test_iter_raw):
		print("Batch %d / %d" % (i, len(test_iter_raw)))
		display.clear_output(wait=True)
		
		YHat = net(X)
		for yhat, yraw in zip(YHat, YRaw):
			yhat = nms(yhat)
			calc.add_image_data(yhat, yraw)

	print("Test VOC mAP:", calc.calculate_VOCmAP())
	print("Test COCO mAP:", calc.calculate_COCOmAP())

	for i in range(20):
		draw_precision_recall(calc.calculate_precision_recall(0.5, i), i)


def test_img(net: torch.nn.Module, src: str):
	"""Test an image

	Args:
		net (torch.nn.Module): network
		src (str): image path
	"""
	net.eval()

	img = cv2.imread(src)
	img = torchvision.transforms.functional.resize(cv2_to_PIL(img), (448, 448))
	to_tensor = torchvision.transforms.ToTensor()
	X = to_tensor(img).unsqueeze_(0)
	YHat = net(X)
	for x, yhat in zip(X, YHat):
		yhat = nms(yhat)
		display.display(cv2_to_PIL(draw_detection_result(tensor_to_cv2(x), yhat, raw=False, thres=0.1)))
