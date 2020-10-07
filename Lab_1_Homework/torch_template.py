from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import Linear_Network
from datetime import datetime


def train(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		data = data.reshape(data.size(0), -1)
		# print(data.shape, target.shape)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % 10 ==0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	# no_grad() prevents codes from tracking records or using memories 
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			data = data.reshape(data.size(0), -1)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction = 'sum').item()
			pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss/=len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))



def main():
	# parser = argparse.ArgumentParser(description = 'Deep Learning Framework 1 Argument')
	# parser.add_argument('--epochs', type = int, default = 100, metavar = 'N', help = 'number of epochs to train and test the model (default=100)')
	# args = parser.parse_args()
	log()

	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
			transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size=128, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size=1000, shuffle=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Linear_Network().to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 1, gamma = 0.8)


	for epoch in range(10):
		train(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)

def log():
	# read file into string
	with open('model.py', 'r') as inputfile:
		textstr = inputfile.read()
		fn = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".txt"
		
		with open("logs/"+fn, 'w') as outputfile:
			outputfile.write(textstr)

if __name__ == "__main__":
	main()