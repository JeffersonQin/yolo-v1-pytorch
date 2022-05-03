from utils import *
from yolo import *
import torch
from torch import nn, tensor
import torchvision
import cv2


model_weight_path = './model/resnet18-pretrained-epoch-145.pth'

resnet18 = torchvision.models.resnet18(pretrained=True)
backbone = nn.Sequential(*list(resnet18.children())[:-2]) # remove avg pool and fc
net = yolo.Yolo(backbone, backbone_out_channels=512)

net.load_state_dict(torch.load(model_weight_path))
net.eval()
net.to('cuda')

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if not ret: break

	img = torchvision.transforms.functional.resize(cv2_to_PIL(frame), (448, 448))
	to_tensor = torchvision.transforms.ToTensor()
	X = to_tensor(img).unsqueeze_(0).to('cuda')
	YHat = net(X)
	for x, yhat in zip(X, YHat):
		yhat = nms(yhat)
		cv2.imshow('Webcam', draw_detection_result(tensor_to_cv2(x), yhat, raw=False, thres=0.1))

	cv2.waitKey(1)
