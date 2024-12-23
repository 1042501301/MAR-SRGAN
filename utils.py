import numpy as np
import torch
import torch.nn as nn
from torch.nn import Softmax
import os
from os import listdir
from os.path import join
import torch.nn.functional as F
from PIL import Image
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.models import vgg19
import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp'])

def calculate_valid_crop_size(crop_size, upscale_factor):
	return crop_size - (crop_size % upscale_factor)

def to_image():
	return Compose([
        ToPILImage(),
        ToTensor()
	])
class TrainDataset(Dataset):
	def __init__(self, dataset_dir, crop_size, upscale_factor):
		super(TrainDataset, self).__init__()
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
		crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
		self.hr_preprocess = Compose([RandomCrop(256), RandomCrop(crop_size), ToTensor()])
		self.lr_preprocess = Compose([ToPILImage(), Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), ToTensor()])

	def __getitem__(self, index):
		hr_image = self.hr_preprocess(Image.open(self.image_filenames[index]))
		lr_image = self.lr_preprocess(hr_image)
		return lr_image, hr_image

	def __len__(self):
		return len(self.image_filenames)
        
class DevDataset(Dataset):
	def __init__(self, dataset_dir, upscale_factor):
		super(DevDataset, self).__init__()
		self.upscale_factor = upscale_factor
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

	def __getitem__(self, index):
		hr_image = Image.open(self.image_filenames[index])
		crop_size = calculate_valid_crop_size(192, self.upscale_factor)
		lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
		hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
		hr_image = RandomCrop(crop_size)(hr_image)
		lr_image = lr_scale(hr_image)
		hr_restore_img = hr_scale(lr_image)
		norm = ToTensor()
		return norm(lr_image), norm(hr_restore_img), norm(hr_image)

	def __len__(self):
		return len(self.image_filenames)

def print_first_parameter(net):	
	for name, param in net.named_parameters():
		if param.requires_grad:
			print (str(name) + ':' + str(param.data[0]))
			return

def check_grads(model, model_name):
	grads = []
	for p in model.parameters():
		if not p.grad is None:
			grads.append(float(p.grad.mean()))

	grads = np.array(grads)
	if grads.any() and grads.mean() > 100:
		print('WARNING!' + model_name + ' gradients mean is over 100.')
		return False
	if grads.any() and grads.max() > 100:
		print('WARNING!' + model_name + ' gradients max is over 100.')
		return False
		
	return True

def get_grads_D(net):
	top = 0
	bottom = 0
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'net.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'net.26.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom
	
def get_grads_D_WAN(net):
	top = 0
	bottom = 0
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'net.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'net.19.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom

def get_grads_G(net):
	top = 0
	bottom = 0
	#torch.set_printoptions(precision=10)
	#torch.set_printoptions(threshold=50000)
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'conv1.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'upsample.2.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom

def vggloss(fake_image,real_image):
	vgg = vgg19(pretrained=True)
	loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
	loss_network = loss_network.cuda()
	mse = nn.MSELoss().cuda()
	with torch.no_grad():
		fake_features = loss_network(fake_image)
		real_features = loss_network(real_image)
	return mse(fake_features,real_features)


def tensor_size(t):
	return t.size()[1] * t.size()[2] * t.size()[3]

def tvloss(fake_image,tv_loss_weight=1):
	x = fake_image
	batch_size = x.size()[0]
	h_x = x.size()[2]
	w_x = x.size()[3]
	count_h = tensor_size(x[:, :, 1:, :])
	count_w = tensor_size(x[:, :, :, 1:])
	h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
	w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
	tvloss = tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
	return torch.mean(tvloss)

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=2):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc1   = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		sigmoid_out = self.sigmoid(out)
		out_c_a = x*sigmoid_out
		return out_c_a

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7,padding=3):
		super(SpatialAttention, self).__init__()
		self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)#16*1*32*32
		max_out, _ = torch.max(x, dim=1, keepdim=True)#16*1*32*32
		x1 = torch.cat([avg_out, max_out], dim=1)#16*2*32*32
		x2 = self.conv1(x1)#16*1*32*32
		x3 = self.sigmoid(x2)
		out_s_a = x*x3
		return out_s_a
class MSCAAttention(nn.Module):
	def __init__(self, dim):
		super(MSCAAttention,self).__init__()
		self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
		self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
		self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

		self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
		self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

		self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
		self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
		self.conv3 = nn.Conv2d(dim, dim, 1)

	def forward(self, x):
		u = x.clone()
		attn = self.conv0(x)

		attn_0 = self.conv0_1(attn)
		attn_0 = self.conv0_2(attn_0)

		attn_1 = self.conv1_1(attn)
		attn_1 = self.conv1_2(attn_1)

		attn_2 = self.conv2_1(attn)
		attn_2 = self.conv2_2(attn_2)

		attn = attn + attn_0 + attn_1 + attn_2

		attn = self.conv3(attn)

		return attn * u

class CombinedAttention(nn.Module):
	def __init__(self, dim):
		super(CombinedAttention, self).__init__()
		self.channel_attn = ChannelAttention(dim)
		self.spatial_attn = SpatialAttention()
		self.msca_attn = MSCAAttention(dim)

	def forward(self, x):
		xc = self.channel_attn(x)
		xs = self.spatial_attn(xc)
		xm = self.msca_attn(x)

		# 使用 xs 和 xm 计算权重
		xs_mean = xs.mean()
		xm_mean = xm.mean()
		weights = F.softmax(torch.tensor([xs_mean, xm_mean]), dim=0)

		return weights[0] * xs + weights[1] * xm

class cs_attention(nn.Module):
	def __init__(self,inplaces,radio=2,kerner_size=7,stride=1,padding=3):
		super(cs_attention,self).__init__()
		self.channelattention = ChannelAttention(inplaces)
		self.spatialattention = SpatialAttention()

	def forward(self, x):
		outc = self.channelattention(x)
		outs = self.spatialattention(outc)
		return  outs


class MS_SSIM_L1_LOSS(nn.Module):
	def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
				 data_range = 1.0,
				 K=(0.01, 0.03),
				 alpha=0.025,
				 compensation=200.0,
				 cuda_dev=0,):
		super(MS_SSIM_L1_LOSS, self).__init__()
		self.DR = data_range
		self.C1 = (K[0] * data_range) ** 2
		self.C2 = (K[1] * data_range) ** 2
		self.pad = int(2 * gaussian_sigmas[-1])
		self.alpha = alpha
		self.compensation=compensation
		filter_size = int(4 * gaussian_sigmas[-1] + 1)
		g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
		for idx, sigma in enumerate(gaussian_sigmas):
			# r0,g0,b0,r1,g1,b1,...,rM,gM,bM
			g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
			g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
			g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
		self.g_masks = g_masks.cuda(cuda_dev)

	def _fspecial_gauss_1d(self, size, sigma):
		coords = torch.arange(size).to(dtype=torch.float)
		coords -= size // 2
		g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
		g /= g.sum()
		return g.reshape(-1)

	def _fspecial_gauss_2d(self, size, sigma):
		gaussian_vec = self._fspecial_gauss_1d(size, sigma)
		return torch.outer(gaussian_vec, gaussian_vec)

	def forward(self, x, y):
		mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
		muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)
		mux2 = mux * mux
		muy2 = muy * muy
		muxy = mux * muy
		sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
		sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
		sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy
		l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)
		cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

		lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
		PIcs = cs.prod(dim=1)

		loss_ms_ssim = 1 - lM*PIcs
		loss_l1 = F.l1_loss(x, y, reduction='none')
		gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
							   groups=3, padding=self.pad).mean(1)
		loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
		loss_mix = self.compensation*loss_mix

		return loss_mix.mean()

