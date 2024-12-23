import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from math import log10
import torchvision.utils as utils
import pytorch_ssim
from model import Generator, Discriminator_WGAN, compute_gradient_penalty
from utils import TrainDataset, DevDataset, to_image, get_grads_G
from utils import tvloss,vggloss,MS_SSIM_L1_LOSS
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
	n_epoch_pretrain = 2
	valing_results = {'mse': 0, 'ssims': 0, 'max_psnr': 0, 'max_ssim': 0, 'batch_sizes': 0,'max_average_psnr': 0, 'max_average_ssim': 0 }
	parser = argparse.ArgumentParser(description='SRGAN Train')
	parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
	parser.add_argument('--num_epochs', default=1800, type=int, help='training epoch')
	parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
	parser.add_argument('--train_set', default='D:/zzz_private_WTY/datasetwty/pc', type=str, help='train set path')
	parser.add_argument('--valid_set', default='D:/zzz_private_WTY/datasetwty/pc1', type=str,help='valid set path')
	parser.add_argument('--check_point', type=int, default=-1, help="continue with previous check_point")
	parser.add_argument('--up_factor', type=int, default=4, help="upscale_factor")
	parser.add_argument('--save_weights_path', type=int, default='save_weights/', help="upscale_factor")
	opt = parser.parse_args()
	input_size = opt.crop_size
	n_epoch = opt.num_epochs
	batch_size = opt.batch_size
	check_point = opt.check_point
	up_factor = opt.up_factor
	save_weights_path = opt.save_weights_path
	wandb.init(project="srgan", save_code=True,config={
		"crop_size": opt.crop_size,
		"num_epochs": opt.num_epochs,
		"batch_size": opt.batch_size,
		"train_set": opt.valid_set,
		"valid_set": opt.train_set,
		"up_factor": opt.up_factor,
		"save_weights":opt.save_weights_path

	})

	if not os.path.exists(save_weights_path):
		os.makedirs(save_weights_path)
	train_set = TrainDataset(opt.train_set, crop_size=input_size, upscale_factor=up_factor)
	train_loader = DataLoader(dataset=train_set, num_workers=3, batch_size=batch_size, shuffle=True,drop_last=True)
	dev_set = DevDataset(opt.valid_set, upscale_factor=up_factor)
	dev_loader = DataLoader(dataset=dev_set, num_workers=2, batch_size=1, shuffle=False)
	mse = nn.MSELoss()
	mssim = MS_SSIM_L1_LOSS()
	netG = Generator()
	print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	netD = Discriminator_WGAN()
	print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

	if torch.cuda.is_available():
		netG.cuda()
		netD.cuda()
		mse.cuda()
	# Pre-train generator using only MSE loss
	if check_point == -1:
		optimizerG = optim.Adam(netG.parameters())
		for epoch in range(1, n_epoch_pretrain + 1):	
			train_bar = tqdm(train_loader)
			netG.train()
			cache1 = {'g_loss': 0}
			for lowres, real_img_hr in train_bar:
				if torch.cuda.is_available():
					real_img_hr = real_img_hr.cuda()
					lowres = lowres.cuda()
				fake_img_hr = netG(lowres)
				netG.zero_grad()
				image_loss = mse(fake_img_hr, real_img_hr)
				cache1['g_loss'] += image_loss
				
				image_loss.backward()
				optimizerG.step()

				train_bar.set_description(desc='[%d/%d] Loss_G: %.4f' % (epoch, n_epoch_pretrain, image_loss))
	optimizerG = optim.Adam(netG.parameters(), lr=1e-5)
	optimizerD = optim.Adam(netD.parameters(), lr=1e-5)
	scheduler = ReduceLROnPlateau(optimizerG, mode='min', factor=0.8, patience=4, min_lr=1e-7)
	if check_point != -1:
		if torch.cuda.is_available():
			netG.load_state_dict(torch.load(f'{save_weights_path}/netG_epoch_' + str(check_point) + '_gpu.pth'))
			netD.load_state_dict(torch.load(f'{save_weights_path}/netD_epoch_' + str(check_point) + '_gpu.pth'))
			optimizerG.load_state_dict(torch.load(f'{save_weights_path}/optimizerG_epoch_' + str(check_point) + '_gpu.pth'))
			optimizerD.load_state_dict(torch.load(f'{save_weights_path}/optimizerD_epoch_' + str(check_point) + '_gpu.pth'))


	for epoch in range(1 + max(check_point, 0), n_epoch + 1 + max(check_point, 0)):
		train_bar = tqdm(train_loader)
		netG.train()
		netD.train()
		cache = {'mse_loss': 0, 'adv_loss': 0, 'tv_loss': 0, 'vgg_loss': 0, 'ssim_loss': 0, 'g_loss': 0, 'd_loss': 0
				 }
		for lowres, real_img_hr in train_bar:
			if torch.cuda.is_available():
				real_img_hr = real_img_hr.cuda()
				lowres = lowres.cuda()
			fake_img_hr = netG(lowres)
			netD.zero_grad()
			logits_real = netD(real_img_hr).mean()
			logits_fake = netD(fake_img_hr).mean()
			gradient_penalty = compute_gradient_penalty(netD, real_img_hr, fake_img_hr)
			d_loss = logits_fake - logits_real + 10 * gradient_penalty
			cache['d_loss'] += d_loss.item()
			d_loss.backward(retain_graph=True)
			optimizerD.step()
			netG.zero_grad()
			image_loss = mse(fake_img_hr, real_img_hr)
			adversarial_loss = -1*netD(fake_img_hr).mean()
			tv_loss = tvloss(fake_img_hr).cuda()
			vgg_loss = vggloss(fake_img_hr,real_img_hr)
			mssim_loss = mssim(fake_img_hr,real_img_hr)
			g_loss = image_loss + 1e-3 * adversarial_loss + 6e-3 * vgg_loss + 2e-8 * tv_loss + 1e-3 * mssim_loss
			cache['mse_loss'] += image_loss.item()
			cache['adv_loss'] += adversarial_loss.item()
			cache['tv_loss']  += tv_loss.item()
			cache['vgg_loss'] += vgg_loss.item()
			cache['g_loss'] += g_loss.item()
			cache['ssim_loss'] += mssim_loss.item()
			g_loss.backward()
			optimizerG.step()
			scheduler.step(image_loss)
			train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f image_loss: %.4f gan_loss: %.4f vgg_loss: %.4f tv_loss: %.4f ssim_loss: %.4f' % (epoch, n_epoch,d_loss, g_loss, image_loss, adversarial_loss,vgg_loss,tv_loss,mssim_loss))

			wandb.log({
				"epoch": epoch,
				"d_loss": d_loss.item(),
				"g_loss": g_loss.item(),
				"image_loss": image_loss.item(),
				"gan_loss": adversarial_loss.item(),
				"vgg_loss": vgg_loss.item(),
				"tv_loss": tv_loss.item(),
				"生成图像得分": logits_fake,
				"真实图像得分": logits_real,
				"梯度惩罚": gradient_penalty
			})


		if torch.cuda.is_available():
			torch.save(netG.state_dict(), '%s/netG_epoch_%d_gpu.pth' % (save_weights_path, epoch))
			if epoch % 5 == 0:
				torch.save(netD.state_dict(), '%s/netD_epoch_%d_gpu.pth' % (save_weights_path, epoch))
				torch.save(optimizerG.state_dict(), '%s/optimizerG_epoch_%d_gpu.pth' % (save_weights_path, epoch))
				torch.save(optimizerD.state_dict(), '%s/optimizerD_epoch_%d_gpu.pth' % (save_weights_path, epoch))

			with torch.no_grad():
				netG.eval()
				dev_bar = tqdm(dev_loader)
				num_loader = len(dev_loader)
				cache2 = {'psnr': 0, 'ssim': 0}
				for val_lr, val_hr_restore, val_hr in dev_bar:
					batch_size = val_lr.size(0)
					lr = val_lr
					hr = val_hr
					if torch.cuda.is_available():
						lr = lr.cuda()
						hr = hr.cuda()
					hsr = netG(lr)
					psnr = 10 * log10(1 / ((hsr - hr) ** 2 + 1e-10).mean().item())
					ssim = pytorch_ssim.ssim(hsr, hr).item()
					dev_bar.set_description(
						desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (psnr, ssim))
					cache2['ssim'] += ssim
					cache2['psnr'] += psnr
				average_ssim = cache2['ssim'] / num_loader
				average_psnr = cache2['psnr'] / num_loader
				if average_psnr > valing_results['max_average_psnr']:
					valing_results['max_average_psnr'] = average_psnr
				if average_ssim > valing_results['max_average_ssim']:
					valing_results['max_average_ssim'] = average_ssim
				wandb.log({
					"val_psnr": average_psnr,
					"val_ssim": average_ssim
				})
				print("当前轮数平均PSNR为：%.4f,平均SSIM为：%.4f" % (average_psnr, average_ssim))
				print("当前最大平均PSNR为：%.4f,平均SSIM为：%.4f" % (valing_results['max_average_psnr'], valing_results['max_average_ssim']))
if __name__ == '__main__':
	main()
