import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import GetCorrectPredCount

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 22
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.ReLU() NEVER!
        ) # output_size = 1 7x7x10 | 7x7x10x10 | 1x1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        
        x = self.pool1(x)
        
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        self.conv1 = nn.Sequential(                  #in_s, out_s, rf_in, rf_out
            nn.Conv2d(1, 8, 3, padding=1),           #28*28, 28*28, 1,3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.Dropout(0.1),
            nn.Conv2d(8, 8, 3, padding=1),           #28*28, 28*28, 3,5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.Dropout(0.1),
            nn.Conv2d(8, 16, 3, padding=1),           #28*28, 28*28, 5,7
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Dropout(0.1),
            # nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),                      #28*28, 14*14, 7,8
            nn.Conv2d(16, 10, 1),                   #14*14, 14*14,, 8,8
                                  
            # nn.Dropout(0.1)
        )
        
        self.conv2 = nn.Sequential(
                                  
            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.Conv2d(10, 12, 3, padding=1),           #14*14, 14*14, 8,12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            
            nn.Dropout(0.1),
            
            nn.Conv2d(12, 12, 3, padding=1),           #14*14, 14*14, 12,16
            nn.BatchNorm2d(12),
            nn.ReLU(),
            
            nn.Dropout(0.1),
            nn.Conv2d(12, 10, 1),  

            nn.MaxPool2d(2, 2),                         #14*14, 7*7, 16,18
            # nn.Dropout(0.1)
            # nn.MaxPool2d(2, 2)
            # nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
                                 #7*7, 7*7, 18,18
            # nn.ReLU(),
            # nn.BatchNorm2d(20),
            nn.Conv2d(10, 14, 3, padding=1),           #7*7, 7*7, 18,26
            nn.BatchNorm2d(14),
            nn.ReLU(),
            
            nn.Conv2d(14, 14, 3, padding=1),           #7*7, 7*7, 26,34
            nn.BatchNorm2d(14),
            nn.ReLU(),
            
            nn.Conv2d(14, 10, 1),                     #7*7, 7*7, 34,34
            # nn.ReLU(),
            nn.AvgPool2d(7)                            #7*7, 1*1, 34,34+6*4
            # nn.AdaptiveMaxPool2d(7)
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
            # nn.MaxPool2d(2, 2)
            # nn.Dropout(0.25)
        )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(16*7*7, 10)
        # )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, 10)
        
        # x = x.view(-1, 16*7*7)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.conv1 = nn.Sequential(                  #in_s, out_s, rf_in, rf_out
            nn.Conv2d(1, 8, 3, padding=0, bias=False),           #28*28, 26*26, 1,3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.Dropout(0.01),

            nn.Conv2d(8, 12, 3, padding=0, bias=False),           #26*26, 24*24, 3,5
            nn.BatchNorm2d(12),
            nn.ReLU(),
            
            nn.Dropout(0.01),

            nn.MaxPool2d(2, 2),                                 #24*24, 12*12, 5,6
            nn.Conv2d(12, 8, 1, bias=False),                   #12*12, 12*12, 6,6

        )

        self.conv2 = nn.Sequential(

            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.Conv2d(8, 12, 3, padding=0, bias=False),           #12*12, 10*10, 6,10
            nn.BatchNorm2d(12),
            nn.ReLU(),
            
            nn.Dropout(0.01),

            nn.Conv2d(12, 16, 3, padding=0, bias=False),           #10*10, 8*8, 10,14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Dropout(0.01),


            nn.MaxPool2d(2, 2),                                     #8*8, 4*4, 14,16
            nn.Conv2d(16, 10, 1, bias=False),                       #4*4, 4*4, 16,16
            # nn.ReLU(),

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 14, 3, padding=1, bias=False),            #4*4, 4*4, 16,24
            nn.BatchNorm2d(14),
            nn.ReLU(),
            
            nn.Dropout(0.01),
            nn.Conv2d(14, 18, 3, padding=1, bias=False),            #4*4, 4*4, 24,32
            nn.BatchNorm2d(18),
            nn.ReLU(),
            
            nn.Dropout(0.01),

            nn.AvgPool2d(4),                                        #4*4, 1*1, 32,44(32+4*3)

            nn.Conv2d(18, 10, 1, bias=False),                       #1*1, 1*1, 44,44

        )



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, 10)

        # x = x.view(-1, 16*7*7)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x




class Net_BN(nn.Module):
    def __init__(self):
        super(Net_BN, self).__init__()

        self.conv1 = nn.Sequential(                  #in_s, out_s, rf_in, rf_out
            nn.Conv2d(3, 16, 3, padding=0, bias=False),           #28*28, 28*28, 1,3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01),

            nn.Conv2d(16, 20, 3, padding=0, bias=False),           #28*28, 28*28, 3,5
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 16, 1, bias=False),
            nn.MaxPool2d(2, 2),
                               #14*14, 14*14, 8,8

        )

        self.conv2 = nn.Sequential(

            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),

            nn.Conv2d(16, 20, 3, padding=1, bias=False),           #14*14, 14*14, 12,16
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 24, 3, padding=1, bias=False),           #14*14, 14*14, 12,16
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.01),

            nn.Conv2d(24, 28, 3, padding=1, bias=False),           #14*14, 14*14, 8,12
            nn.ReLU(),
            nn.BatchNorm2d(28),
            nn.Dropout(0.01),

            nn.Conv2d(28, 16, 1, bias=False),
            nn.MaxPool2d(2, 2),

            # nn.ReLU(),

        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 20, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.01),

            nn.Conv2d(24, 28, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(28),
            nn.Dropout(0.01),

            nn.AvgPool2d(4),

            nn.Conv2d(28, 10, 1, bias=False),

        )



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, 10)

        # x = x.view(-1, 16*7*7)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net_LN(nn.Module):
    def __init__(self):
        super(Net_LN, self).__init__()

        self.conv1 = nn.Sequential(                  #in_s, out_s, rf_in, rf_out
            nn.Conv2d(3, 16, 3, padding=0, bias=False),           #28*28, 28*28, 1,3
            nn.ReLU(),
            nn.GroupNorm(1,16),
            nn.Dropout(0.01),

            nn.Conv2d(16, 20, 3, padding=0, bias=False),           #28*28, 28*28, 3,5
            nn.ReLU(),
            nn.GroupNorm(1,20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 16, 1, bias=False),
            nn.MaxPool2d(2, 2),
                               #14*14, 14*14, 8,8

        )

        self.conv2 = nn.Sequential(

            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),

            nn.Conv2d(16, 20, 3, padding=1, bias=False),           #14*14, 14*14, 12,16
            nn.ReLU(),
            nn.GroupNorm(1,20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 24, 3, padding=1, bias=False),           #14*14, 14*14, 12,16
            nn.ReLU(),
            nn.GroupNorm(1,24),
            nn.Dropout(0.01),

            nn.Conv2d(24, 28, 3, padding=1, bias=False),           #14*14, 14*14, 8,12
            nn.ReLU(),
            nn.GroupNorm(1,28),
            nn.Dropout(0.01),

            nn.Conv2d(28, 16, 1, bias=False),
            nn.MaxPool2d(2, 2),

            # nn.ReLU(),

        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 20, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,24),
            nn.Dropout(0.01),

            nn.Conv2d(24, 28, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,28),
            nn.Dropout(0.01),

            nn.AvgPool2d(4),

            nn.Conv2d(28, 10, 1, bias=False),

        )



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, 10)

        # x = x.view(-1, 16*7*7)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class Net_GN(nn.Module):
    def __init__(self):
        super(Net_GN, self).__init__()

        self.conv1 = nn.Sequential(                  #in_s, out_s, rf_in, rf_out
            nn.Conv2d(3, 16, 3, padding=0, bias=False),           #28*28, 28*28, 1,3
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout(0.01),

            nn.Conv2d(16, 20, 3, padding=0, bias=False),           #28*28, 28*28, 3,5
            nn.ReLU(),
            nn.GroupNorm(4,20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 16, 1, bias=False),
            nn.MaxPool2d(2, 2),
                               #14*14, 14*14, 8,8

        )

        self.conv2 = nn.Sequential(

            # nn.ReLU(), -->why relu. aggregators dont need relu to be appplied
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.1),

            nn.Conv2d(16, 20, 3, padding=1, bias=False),           #14*14, 14*14, 12,16
            nn.ReLU(),
            nn.GroupNorm(4,20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 24, 3, padding=1, bias=False),           #14*14, 14*14, 12,16
            nn.ReLU(),
            nn.GroupNorm(4,24),
            nn.Dropout(0.01),

            nn.Conv2d(24, 28, 3, padding=1, bias=False),           #14*14, 14*14, 8,12
            nn.ReLU(),
            nn.GroupNorm(4,28),
            nn.Dropout(0.01),

            nn.Conv2d(28, 16, 1, bias=False),
            nn.MaxPool2d(2, 2),

            # nn.ReLU(),

        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 20, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,20),
            nn.Dropout(0.01),

            nn.Conv2d(20, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,24),
            nn.Dropout(0.01),

            nn.Conv2d(24, 28, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,28),
            nn.Dropout(0.01),

            nn.AvgPool2d(4),

            nn.Conv2d(28, 10, 1, bias=False),

        )



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, 10)

        # x = x.view(-1, 16*7*7)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



def model_summary(model, input_size):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    summary(model, input_size=input_size)


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss=0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    if scheduler!=None:
        scheduler.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc.append(100*correct/processed)
  train_losses.append(train_loss)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))  

def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")