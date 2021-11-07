from torchvision.utils import save_image
import os
import numpy as np

## Helper Functions for checking Gpu, Counting params, Plotting results

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_results(train_loader):

	for i, (img, lab) in enumerate(train_loader):
		print(img.size())
		tmp = torch.squeeze(img[0]).transpose(0,1)
		tmp = tmp.transpose(1, 2)
		
		save_image(img[0], path+'orig1.png')
		print(tmp.size())
		img = img*255
		lab = lab*255
		img = img.cuda()
		img = F.interpolate(img, (H, W))
		lab = F.interpolate(lab, (H, W))
		print(lab.size())
		lab = lab.reshape(lab.size()[0], H, W).long().cuda()
		outputs = model(img)
		print(outputs.size(), lab.size())
		acc = torch.sum(outputs.argmax(1)==lab) / (512*256)
		print('acc',acc)

		outputs = outputs[0].argmax(0).reshape(512, 256).cpu().detach().numpy()
		lab = lab[0].reshape(512, 256).cpu().detach().numpy()

		plt.rcParams["figure.figsize"] = (21,9)

		fig, (ax2, ax3, ax1) = plt.subplots(1, 3, constrained_layout = True)

		ax1.set_title('Predictions',  fontsize=24)
		ax1.tick_params(left = False, right = False , labelleft = False ,
					labelbottom = False, bottom = False)
		ax1.imshow(outputs, interpolation='nearest')
		ax1.set_aspect('auto')
		ax2.set_title('Original', fontsize=24)
		ax2.tick_params(left = False, right = False , labelleft = False ,
					labelbottom = False, bottom = False)
		ax2.imshow(tmp.reshape(H,W, 3).cpu().detach().numpy(), interpolation='nearest')
		ax2.set_aspect('auto')
		ax3.set_title('Ground Truth',  fontsize=24)
		ax3.tick_params(left = False, right = False , labelleft = False ,
					labelbottom = False, bottom = False)
		ax3.imshow(lab, interpolation='nearest')
		ax3.set_aspect('auto')
		plt.savefig(path+'combined.png', bbox_inches='tight')

		plt.imsave(path+'pred_image1.png', outputs)
		plt.imsave(path+'gt_image1.png',lab)
		break