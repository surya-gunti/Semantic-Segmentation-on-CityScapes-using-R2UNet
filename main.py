import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from network import R2U_Net
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from metrics import confusion_mat, calc_sensitivity, calc_specificity,calc_weights, calc_dice, calc_jaccord, calc_accuracy
from helper import plot_results, get_free_gpu, count_parameters
import matplotlib.pyplot as plt
import os



free_gpu_id = get_free_gpu()

torch.cuda.set_device(int(free_gpu_id))


##Baseline R2UNet Layers list
layers_list=[64, 128, 256, 512, 1024]


model = R2U_Net(output_ch=34, t=2, layers_list=layers_list)
model.cuda()



bs = 4
epochs = 120
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
H = 512
W = 256
n_classes = 34
path = "/src/project/"




train_dataset = torchvision.datasets.Cityscapes('./cityscapes/', split='train', mode='fine', transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((H,W)),]),
                     target_transform= transforms.Compose([transforms.ToTensor(), ]), target_type='semantic')

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True )

val_dataset = torchvision.datasets.Cityscapes('./cityscapes/', split='val', mode='fine', transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((H,W)),]),
                     target_transform= transforms.Compose([transforms.ToTensor(), ]), target_type='semantic')

val_loader = DataLoader(val_dataset, batch_size=bs  )



# loss function
loss_f = nn.CrossEntropyLoss()

# optimizer variable
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

train_losses, acc_scores, sensitivity_scores, specificity_scores, dice_scores, jaccord_scores = [], [], [], [], [], []
val_losses, val_acc_scores, val_sensitivity_scores, val_specificity_scores, val_dice_scores, val_jaccord_scores = [], [], [], [], [], []

s = set()
vs = set()

##Training the Model
for e in range(epochs):
	conf_mat = np.zeros((n_classes, n_classes))
	running_correct, running_total = 0, 0
	for i, (img, lab) in enumerate(train_loader):
		#print(img.size(), lab.size())
		opt.zero_grad()
		img = img*255
		lab = lab*255
		img = F.interpolate(img, (H, W))
		lab = F.interpolate(lab, (H, W))
		s.update(torch.unique(lab).cpu().detach().numpy())
		img = img.cuda()
		lab = lab.reshape(lab.size()[0], H, W).long().cuda()
		outputs = model(img)
		loss = loss_f(outputs, lab)
		loss.backward()
		opt.step()

		## Calculating Confusion Matrices for each iteration and updating the global confusion matrix variable
		mat = confusion_mat(lab.reshape(-1).cpu().detach().numpy(), outputs.argmax(1).reshape(-1).cpu().detach().numpy())
		conf_mat+=mat

		##Calculating Overall Accuracy
		correct = torch.sum(outputs.argmax(1)==lab)
		total = torch.numel(lab)

		running_correct+=correct
		running_total+=total
		
		if i % 10 == 0:
		    print("epoch{}, iter{}, Training loss: {}".format(e, i, loss.data))
	
	## Calucating Metrics on Training Data
	sensitivity, non_zero_sens = calc_sensitivity(conf_mat)
	weights = calc_weights(conf_mat)
	weights = np.nan_to_num(weights)
	weighted_sens = sensitivity*weights
	print("Weighted Sensitivity: {}".format(weighted_sens.sum()))
	sensitivity_scores.append(weighted_sens.sum())

	specificity = calc_specificity(conf_mat)
	weighted_spec = specificity * weights
	print("Weighted Specificity: {}".format(weighted_spec.sum()))
	specificity_scores.append(weighted_spec.sum())

	dice = calc_dice(conf_mat)
	weighted_dice = dice * weights
	print("Weighted Dice Score: {}, {}".format(weighted_dice.sum(), dice.mean()))
	dice_scores.append(weighted_dice.sum())

	jaccord = calc_jaccord(conf_mat)
	weighted_jaccord = jaccord * weights
	print("Weighted Jaccord Score: {}, {}".format(weighted_jaccord.sum(), jaccord.mean()))
	jaccord_scores.append(weighted_jaccord.sum())

	train_acc = float(running_correct)/running_total
	print("epoch{}, Training loss: {}, Training Accuracy: {}".format(e, loss.item(), train_acc ))
	train_losses.append(loss.item())
	acc_scores.append(train_acc)

	## Saving the Model after each epoch
	torch.save(model.cpu().state_dict(), path+'models/model_epoch_'+str(e)+'.pth')
	model.cuda()

	## Calculating the Performace on Validation Dataset
	val_running_correct, val_running_total = 0, 0
	val_conf_mat = np.zeros((n_classes, n_classes))
	with torch.no_grad():
		for i, (img, lab) in enumerate(val_loader):
			img = img*255
			lab = lab*255
			img = img.cuda()
			img = F.interpolate(img, (H, W))
			lab = F.interpolate(lab, (H, W))
			vs.update(torch.unique(lab).cpu().detach().numpy())			
			lab = lab.reshape(lab.size()[0], H, W).long().cuda()
			outputs = model(img)
			val_loss = loss_f(outputs, lab)

			correct = torch.sum(outputs.argmax(1)==lab)
			total = torch.numel(lab)
			val_running_correct+=correct
			val_running_total+=total
			mat = confusion_mat(lab.reshape(-1).cpu().detach().numpy(), outputs.argmax(1).reshape(-1).cpu().detach().numpy())
			val_conf_mat+=mat
			# val_acc = float(val_running_correct)/val_running_total

		#print("validation Uniques", vs)
		sensitivity, non_zero_sens = calc_sensitivity(val_conf_mat)
		weights = calc_weights(val_conf_mat)
		weights = np.nan_to_num(weights)

		weighted_sens = sensitivity*weights
		print("Validation Weighted Sensitivity", weighted_sens.sum())
		val_sensitivity_scores.append(weighted_sens.sum())

		specificity = calc_specificity(val_conf_mat)
		weighted_spec = specificity * weights
		print("Validation Weighted Specificity: {}".format(weighted_spec.sum()))
		val_specificity_scores.append(weighted_spec.sum())

		dice = calc_dice(val_conf_mat)
		weighted_dice = dice * weights
		print("Validation Weighted Dice Score: {}, {}".format(weighted_dice.sum(), dice.mean()))
		val_dice_scores.append(weighted_dice.sum())

		jaccord = calc_jaccord(val_conf_mat)
		weighted_jaccord = jaccord * weights
		print("Validation Weighted Jaccord Score: {}, {}".format(weighted_jaccord.sum(), jaccord.mean()))
		val_jaccord_scores.append(weighted_jaccord.sum())

		val_acc = calc_accuracy(val_conf_mat)
		print("epoch{},  Validation Accuracy: {}".format(e, val_acc ))
		val_losses.append(val_loss.item())
		val_acc_scores.append(val_acc)

	print("epoch{}, Validation loss: {}, Validation Accuracy: {}".format(e, val_loss.data, val_acc ))

val_losses, val_acc_scores, val_sensitivity_scores, val_specificity_scores, val_dice_scores, val_jaccord_scores
print("Training Losses", train_losses)
print("Training Accuracies", acc_scores)
print("Training Sensitivity",sensitivity_scores)
print("Training Specificity",specificity_scores)
print("Training Dice", dice_scores)
print("Training Jaccord", jaccord_scores)

print("Validation Losses", val_losses)
print("Validation Accuracies", val_acc_scores)
print("Validation Sensitivity",val_sensitivity_scores)
print("Validation Specificity",val_specificity_scores)
print("Validation Dice", val_dice_scores)
print("Validation Jaccord", val_jaccord_scores)


## Plotting the results
plot_results(model, val_loader)





