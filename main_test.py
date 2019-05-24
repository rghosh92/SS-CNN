# from RadialHarmonic_Network import *
import torchvision.transforms as transforms
import torch.optim as optim
from ScaleSteerableInvariant_Network import *
# from ScaleInvNet import *
# from ScaleqNet import *
# from ScaleSteerableEquivariant_Network import *
# from spatial_transformer import st_net
import os,pickle
import numpy as np
import torch
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import scipy.misc


import time
import scipy.misc

# This is the testbench for the
# MNIST-Scale, FMNIST-Scale and CIFAR-10-Scale datasets.
# The networks and network architecture are defiend
# within their respective libraries


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, inputs, labels,transform=None):
        'Initialization'
        self.labels = labels
        # self.list_IDs = list_IDs
        self.inputs = inputs
        self.transform = transform
        self.dataset_name = dataset_name


    def __len__(self):
        'Denotes the total number of samples'
        return self.inputs.shape[0]

    def cutout(self,img,x,y,size):
        size = int(size/2)
        lx = np.maximum(0,x-size)
        rx = np.minimum(img.shape[0],x+size)
        ly = np.maximum(0, y - size)
        ry = np.minimum(img.shape[1], y + size)

        img[lx:rx,ly:ry,:] = 0
        return img

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        img = self.inputs[index]
        if self.dataset_name == 'STL10':
            img = np.transpose(img, [1, 2, 0])

        # Cutout module begins
        # xcm = int(np.random.rand()*95)
        # ycm = int(np.random.rand()*95)
        # img = self.cutout(img,xcm,ycm,24)
        #Cutout module ends

        # print(np.max(img),np.min(img))

        img = np.float32(scipy.misc.imresize(img,2.0))

        # Optional:
        # img = img / np.max(img)



        if self.transform is not None:
            img = self.transform(img)

        y = int(self.labels[index])

        return img, y



def load_dataset(dataset_name,val_splits,training_size):

    os.chdir(dataset_name)
    a = os.listdir()
    listdict = []

    # meanarr = [0.4914, 0.4822, 0.4465]
    # stdarr = [0.247,0.243,0.261]

    for split in range(val_splits):

        listdict.append(pickle.load(open(a[split], 'rb')))

        listdict[-1]['train_data'] = np.float32(listdict[-1]['train_data'][0:training_size, :, :])
        listdict[-1]['train_label'] = listdict[-1]['train_label'][0:training_size]

        listdict[-1]['test_data'] = np.float32(listdict[-1]['test_data'])

        listdict[-1]['test_label'] = np.float32(listdict[-1]['test_label'])

    os.chdir('..')

    return listdict


def train_network(net,trainloader,init_rate, step_size,gamma,total_epochs,weight_decay):

    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(),lr=init_rate, momentum=0.9,weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    net = net.cuda()
    net = net.train()

    # s = time.time()

    for epoch in range(total_epochs):
        #
        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        torch.cuda.empty_cache()
        scheduler.step()
        print(epoch)

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    net = net.eval()
    return net


def test_network(net,testloader,test_labels):

    net = net.eval()
    correct = torch.tensor(0)
    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        correct = correct + torch.sum(predicted == labels)
        torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy



if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_name = 'MNIST'
    val_splits = 1

    # Good result on MNIST-Scale 1000 Training
    # training_size = 1000
    # batch_size = 100
    # init_rate = 0.05
    # weight_decay = 0.06

    training_size = 10000
    batch_size = 400
    init_rate = 0.04
    decay_normal = 0.04
    decay_special = 0.04

    step_size = 10

    gamma = 0.7
    total_epochs = 10


    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         ])

    Networks_to_train = [Net_steerinvariant_mnist_scale()]

    listdict = load_dataset(dataset_name, val_splits, training_size)
    accuracy_all = np.zeros((val_splits,len(Networks_to_train)))


    for i in range(val_splits):

        Networks_to_train = [Net_steerinvariant_mnist_scale()]

        train_data = listdict[i]['train_data']
        train_labels = listdict[i]['train_label']
        test_data = listdict[i]['test_data']
        test_labels = listdict[i]['test_label']

        Data_train = Dataset(dataset_name,train_data,train_labels,transform_train)
        Data_test = Dataset(dataset_name, test_data, test_labels, transform_test)

        trainloader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        testloader = torch.utils.data.DataLoader(Data_test, batch_size=int(len(test_labels)/200),
                                                  shuffle=False, num_workers=2)


        for j in range(len(Networks_to_train)):
            net = train_network(Networks_to_train[j],trainloader, init_rate, step_size,gamma,total_epochs,decay_normal)
            accuracy = test_network(net,testloader,test_labels)
            accuracy_train = test_network(net,trainloader,train_labels)
            torch.save(net.state_dict(),'SSCNN_mnistlocal_latest'+str(i)+'.pt')
            print("Train:",accuracy_train,"Test:",accuracy)
            accuracy_all[i,j] = accuracy


    print("Mean Accuracies of Networks:", np.mean(accuracy_all,0))
    print("Standard Deviations of Networks:",np.std(accuracy_all,0))
    print(accuracy_all)


