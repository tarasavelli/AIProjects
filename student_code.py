# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):

    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions

        ## 32 x 32

        ## Layer 1
        self.convLayer1 = torch.nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0)
        self.relu1 = torch.nn.ReLU()
        self.mxPool1 = torch.nn.MaxPool2d(kernel_size = 2, padding = 0, stride = 2)

        ## 14 x 14 x 6

        ## Layer 2
        self.convLayer2 = torch.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)
        self.relu2 = torch.nn.ReLU()
        self.mxPool2 = torch.nn.MaxPool2d(kernel_size = 2, padding = 0, stride = 2) 

        ## 5 x 5 x 16

        ## Layer 3
        self.flattenLayer = torch.nn.Flatten()

        ## 400 x 1

        ## Layer 4
        self.linear1 = torch.nn.Linear(in_features = 400, out_features = 256)
        self.relu3 = torch.nn.ReLU()

        ## 256 x 1

        ## Layer 5
        self.linear2 = torch.nn.Linear(in_features = 256, out_features = 128)
        self.relu4 = torch.nn.ReLU()

        ## 128 x 1

        ## Layer 6
        self.linear3 = torch.nn.Linear(in_features = 128, out_features = num_classes)

        ## Output: 100 x 1

        



    def forward(self, x):
        shape_dict = {}
        # certain operations

        batchSize = len(x)

        ## Layer 1
        x = self.convLayer1(x)
        x = self.relu1(x)
        x = self.mxPool1(x)

        shape = [batchSize, 6, 14, 14]
        shape_dict.update({1 : list(x.shape)})

        

        ## Layer 2
        x = self.convLayer2(x)
        x = self.relu2(x)
        x = self.mxPool2(x)

        shape = [batchSize, 16, 5, 5]
        shape_dict.update({2 : list(x.shape)})

        ## Layer 3
        x = self.flattenLayer(x)

        shape = [batchSize, 400]
        shape_dict.update({3 : list(x.shape)})

        ## Layer 4
        x = self.linear1(x)
        x = self.relu3(x)

        shape = [batchSize, 256]
        shape_dict.update({4 : list(x.shape)})

        ## Layer 4
        x = self.linear2(x)
        x = self.relu4(x)

        shape = [batchSize, 128]
        shape_dict.update({5 : list(x.shape)})

        ## Layer 5
        x = self.linear3(x)

        shape = [batchSize, 100]
        shape_dict.update({6 : list(x.shape)})

        out = x
        
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    for name, param in model.named_parameters():
        model_params += param.numel()

    

    ##convLayer1 = (model.convLayer1.kernel_size[0] * model.convLayer1.kernel_size[1] * model.convLayer1.in_channels + 1) * model.convLayer1.out_channels
    ##convLayer2 = (model.convLayer2.kernel_size[0] * model.convLayer2.kernel_size[1] * model.convLayer2.in_channels + 1) * model.convLayer2.out_channels 
    ##linear1 = model.linear1.in_features * model.linear1.out_features + model.linear1.out_features
    ##linear2 = model.linear2.in_features * model.linear2.out_features + model.linear2.out_features 
    ##linear3 = model.linear3.in_features * model.linear3.out_features + model.linear3.out_features 

    return model_params / 1000000


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

if __name__ == "__main__":
    model = LeNet()