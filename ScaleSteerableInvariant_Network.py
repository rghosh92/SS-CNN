import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils import *
import torch.nn.functional
import numpy as np
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import torch.optim as optim
# from scipy import signal
# import torchvision
# from scipy.ndimage.filters import gaussian_filter
# import torchvision.transforms as transforms
# import sys
# from copy import copy
# import pickle
# from torch.nn.modules.utils import _pair, _quadruple
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from scale_steering_lite import *
# from torchvision.transforms import *
# sys.path.append('../../Libraries')
from torch.optim.lr_scheduler import StepLR
# from invariance_estimation import *

import os


class steerable_conv(nn.Module):


    def __init__(self,kernel_size,in_channels, out_channels, k_range = [2], phi_range = np.linspace(-np.pi,np.pi,9),
                 sigma_phi_range = [np.pi/8],ker_size_range=np.arange(3,15,2), phase_range = [0,np.pi/2], basis_scale=[1.0],drop_rate=1.0):
        super(steerable_conv,self).__init__()

        basis_size = len(phi_range) * len(sigma_phi_range) * len(phase_range) * len(basis_scale)
        self.mult_real = Parameter(torch.Tensor(len(k_range),out_channels, in_channels, basis_size))
        self.mult_imag = Parameter(torch.Tensor(len(k_range),out_channels, in_channels, basis_size))

        self.num_scales = len(ker_size_range)
        self.scale_range = np.ones(self.num_scales)

        for i in range(self.num_scales):
            self.scale_range[i] = ker_size_range[i]/kernel_size[0]

        self.ker_size_range = ker_size_range

        max_size = self.ker_size_range[-1]

        self.filter_real = torch.zeros(len(k_range), max_size, max_size, basis_size)
        self.filter_imag = torch.zeros(len(k_range), max_size, max_size, basis_size)
        self.filter_real.requires_grad = False
        self.filter_imag.requires_grad = False
        self.greedy_multiplier = 1
        self.k_range = k_range


        self.max_size = max_size
        self.const_real = Parameter(torch.Tensor(out_channels,in_channels))
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.basis_size = basis_size
        self.kernel_size = kernel_size
        self.effective_k = np.zeros(len(k_range))


        self.init_he = torch.zeros(len(k_range),basis_size)

        for i in range(len(k_range)):
            count = 0
            for j in range(len(phi_range)):
                for k in range(len(sigma_phi_range)):
                    for p in range(len(phase_range)):
                        for b in range(len(basis_scale)):
                            filter_real, filter_imag,eff_k = generate_filter_basis([max_size,max_size],
                                                                             phi_range[j], sigma_phi_range[k],
                                                                             k_range[i], basis_scale[b], phase_range[p],drop_rate)
                            filter_real = filter_real/(np.linalg.norm(filter_real))
                            filter_imag = filter_imag/(np.linalg.norm(filter_imag))
                            self.effective_k[i] = eff_k

                            self.init_he[i,count] = 2 / (
                                        basis_size *in_channels*out_channels* torch.pow(torch.norm(torch.from_numpy(filter_real)), 2.0))
                            self.filter_real[i,:, :, count] = torch.from_numpy(filter_real)
                            self.filter_imag[i,:, :, count] = torch.from_numpy(filter_imag)
                            count = count + 1

        # self.power_law_mult = np.power((self.effective_k),-0.5)

        self.reset_parameters()

    def combination(self):

        # weight1 = torch.zeros(self.mult_real.shape[0], self.mult_real.shape[1],
        #                       self.filter_real.shape[0], self.filter_real.shape[1])
        W_all = []
        # W_imag_all = []

        Smid = int((self.max_size-1)/2)

        # Below: Whether to use all filter orders at all scales or not
        k_num_scales = np.ones(self.num_scales)*len(self.k_range)

        # k_num_scales[self.scale_range<1.3] = len(self.k_range) - 1
        # k_num_scales[self.scale_range <= 0.8] = len(self.k_range) - 2
        # k_num_scales[self.scale_range >= 1.3] = len(self.k_range)


        for i in range(self.num_scales):
            s = self.scale_range[i]
            Swid = int((self.ker_size_range[i]-1)/2)
            W_real = torch.zeros(len(self.k_range),self.out_channels,self.in_channels,self.ker_size_range[i],self.ker_size_range[i])
            W_imag = torch.zeros(len(self.k_range),self.out_channels,self.in_channels,self.ker_size_range[i],self.ker_size_range[i])

            # W_real = torch.zeros(len(self.k_range), self.out_channels, self.in_channels, self.max_size,
            #                      self.max_size)
            # W_imag = torch.zeros(len(self.k_range), self.out_channels, self.in_channels, self.max_size,
            #                      self.max_size)

            mul =1
            #

            for k in range(int(k_num_scales[i])):
                k_val = self.effective_k[k]
                mult_real_k = self.mult_real[k,:,:,:]*np.cos(-k_val*np.log(s)) - self.mult_imag[k,:,:,:]*np.sin(-k_val*np.log(s))
                mult_imag_k = self.mult_real[k,:,:,:]*np.sin(-k_val*np.log(s)) + self.mult_imag[k,:,:,:]*np.cos(-k_val*np.log(s))
                W_real[k,:,:,:,:] = torch.einsum("ijk,abk->ijab", mult_real_k,self.filter_real[k,Smid-Swid:Smid+Swid+1,Smid-Swid:Smid+Swid+1,:]).contiguous()
                W_imag[k,:,:,:,:] = torch.einsum("ijk,abk->ijab", mult_imag_k,self.filter_imag[k,Smid-Swid:Smid+Swid+1,Smid-Swid:Smid+Swid+1,:]).contiguous()
                # W_real[k, :, :, :, :] = torch.einsum("ijk,abk->ijab", mult_real_k,
                #                                      mul * self.filter_real[k, :,:, :]).contiguous()
                # W_imag[k, :, :, :, :] = torch.einsum("ijk,abk->ijab", mult_imag_k,
                #                                      mul * self.filter_imag[k,:,:, :]).contiguous()

            W_final = torch.sum(W_real,0)-torch.sum(W_imag,0)
            W_all.append(W_final)

        return W_all

        # print(W_real.shape)
        # self.const_mat = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        # self.const_mat[:, :, int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[1] - 1) / 2)] \
        #     = torch.mul(self.const_real, torch.ones(self.out_channels, self.in_channels))


        # return W_real - W_imag



    def forward(self):
        return self.combination()

    def reset_parameters(self):

        # he =  0.2 / basis_size
        self.const_real.data.uniform_(-0.00001, 0.00001)

        for i in range(self.mult_real.shape[3]):
            for k in range(len(self.k_range)):
                self.mult_real[k,:, :, i].data.uniform_(-torch.sqrt(self.init_he[k,i]), torch.sqrt(self.init_he[k,i]))
                self.mult_imag[k,:, :, i].data.uniform_(-torch.sqrt(self.init_he[k,i]), torch.sqrt(self.init_he[k,i]))


class ScaleConv_steering(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,n_scales_small=5, n_scales_big=3, mode=1, angle_range=120,
                 k_range = [0.5, 1, 2], phi_range = np.linspace(0,np.pi,9),
                 sigma_phi_range = [np.pi/16], ker_size_range = np.arange(3,17,2), phase_range = [-np.pi/4], basis_scale=[1.0],drop_rate=1.0 ):

        super(ScaleConv_steering, self).__init__()

        # kernel_size = ntuple(2)(kernel_size)
        # stride = ntuple(2)(stride)
        # padding = ntuple(2)(padding)
        # dilation = ntuple(2)(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ker_size_range = ker_size_range

        self.n_scales_small = n_scales_small
        self.n_scales_big = n_scales_big
        self.n_scales = n_scales_small + n_scales_big
        self.angle_range = angle_range
        self.mode = mode
        # Angles
        self.angles = np.linspace(-angle_range * self.n_scales_small / self.n_scales,
                                  angle_range * self.n_scales_big / self.n_scales, self.n_scales, endpoint=True)


        # If input is vector field, we have two filters (one for each component)
        self.steer_conv = steerable_conv(self.kernel_size, in_channels,out_channels,k_range, phi_range,
                            sigma_phi_range,ker_size_range,phase_range, basis_scale,drop_rate)

        # self.bias = Parameter(torch.Tensor(1, self.out_channels, 1, 1))
        # self.bias.data.fill_(0)
        # self.bias.requires_grad = True

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.mode == 2:
            self.weight1.data.uniform_(-stdv, stdv)
            self.weight2.data.uniform_(-stdv, stdv)
            # for i in range(self.mult_real.shape[2]):
            #     self.mult_real[:,:,i].data.uniform_(-torch.sqrt(self.init_he[i]),torch.sqrt(self.init_he[i]))
            #     self.mult_imag[:,:,i].data.uniform_(-torch.sqrt(self.init_he[i]),torch.sqrt(self.init_he[i]))


    def _apply(self, func):
        # This is called whenever user calls model.cuda()
        # We intersect to replace tensors and variables with cuda-versions
        super(ScaleConv_steering, self)._apply(func)



    def forward(self, input):
        # if global_epoch==1 and self.mode==2:
        #     # self.scale_noise =
        #     self.scale_noise = torch.zeros((input[0].shape[2], input[0].shape[3]), dtype=torch.float64, requires_grad=True)
        #     self.scale_noise.data.uniform_(-0.001,0.001)

        outputs = []
        orig_size = list(input.data.shape[2:4])

        self.weight_all = self.steer_conv()

        # Input upsampling scales (smaller filter scales)
        # input_s = input.clone()

        for i in range(len(self.weight_all)):
            padding = int((self.ker_size_range[i]-1)/2)
            out = F.conv2d(input, self.weight_all[i], None, self.stride, padding, self.dilation)
            outputs.append(out.unsqueeze(-1))

        strength, _ = torch.max(torch.cat(outputs, -1), -1)



        # B = self.bias.repeat(strength.shape[0], 1, strength.shape[2], strength.shape[3])

        return F.relu(strength)



# def grad_update(angles,responses):
#     all_outs = torch.cat(responses, 4)


class Net_steerinvariant_mnistlocal_scale(nn.Module):
    def __init__(self):
        super(Net_steerinvariant_mnistlocal_scale, self).__init__()


        lays = [12, 32, 48]
        kernel_sizes = [13, 13, 13]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        # Good configuration saved
        #
        # self.conv1 = ScaleConv_steering(1, 20, [kernel_sizes[0], kernel_sizes[0]], 1,
        #                                 padding=pads[0], sigma_phi_range=[np.pi/16],
        #                                  mode=1)
        # self.conv2 = ScaleConv_steering(20, 50, [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi/16],
        #                                 basis_scale = [0.2], mode=1)
        # self.conv3 = ScaleConv_steering(50, 100, [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi / 16],
        #                                 basis_scale = [0.1], mode=1)

        # For less data size
        lays = [30,60,90]

        # For 100 percent data size
        # lays = [30,60,90]

        self.conv1 = ScaleConv_steering(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                        padding=pads[0], sigma_phi_range=[np.pi / 16],
                                        k_range = [0.5,1,2], ker_size_range=np.arange(7,21,2),
                                        # stride = 2,
                                        phi_range = np.linspace(0, np.pi, 9),
                                        phase_range = [-np.pi/4],
                                        mode=1)
        self.conv2 = ScaleConv_steering(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        ker_size_range=np.arange(7,21,2),
                                        # stride=2,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi/16],
                                        mode=1,drop_rate=2)
        self.conv3 = ScaleConv_steering(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7,21,2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1,drop_rate=4)


        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(lays[0])
        self.pool2 = nn.MaxPool2d(2,padding=(0,1))
        self.bn2 = nn.BatchNorm2d(lays[1])
        self.pool3 = nn.MaxPool2d(kernel_size=(8,5),padding=(2,0))
        self.bn3 = nn.BatchNorm2d(lays[2])
        self.bn3_mag = nn.BatchNorm2d(lays[2])
        self.fc1 = nn.Conv2d(lays[2]*8, 512, 1)
        self.fc1bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(512, 10, 1)  # FC2

    def forward(self, x):
        # if self.orderpaths == True:

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        xm = self.pool3(x)
        xm = self.bn3_mag(xm)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm

class Net_steerinvariant_mnist_scale(nn.Module):
    def __init__(self):
        super(Net_steerinvariant_mnist_scale, self).__init__()


        lays = [12, 32, 48]
        kernel_sizes = [11, 11, 11]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        # Good configuration saved
        #
        # self.conv1 = ScaleConv_steering(1, 20, [kernel_sizes[0], kernel_sizes[0]], 1,
        #                                 padding=pads[0], sigma_phi_range=[np.pi/16],
        #                                  mode=1)
        # self.conv2 = ScaleConv_steering(20, 50, [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi/16],
        #                                 basis_scale = [0.2], mode=1)
        # self.conv3 = ScaleConv_steering(50, 100, [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi / 16],
        #                                 basis_scale = [0.1], mode=1)

        # For less data size
        lays = [30,60,90]

        # For 100 percent data size
        # lays = [30,60,90]

        self.conv1 = ScaleConv_steering(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                        padding=pads[0], sigma_phi_range=[np.pi / 16],
                                        k_range = [0.5,1,2], ker_size_range=np.arange(7,19,2),
                                        # stride = 2,
                                        phi_range = np.linspace(0, np.pi, 9),
                                        phase_range = [-np.pi/4],
                                        mode=1)
        self.conv2 = ScaleConv_steering(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        ker_size_range=np.arange(7,19,2),
                                        # stride=2,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi/16],
                                        mode=1,drop_rate=2)
        self.conv3 = ScaleConv_steering(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7,19,2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1,drop_rate=4)


        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(lays[0])
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(lays[1])
        self.pool3 = nn.MaxPool2d(8,padding=2)
        self.bn3 = nn.BatchNorm2d(lays[2])
        self.bn3_mag = nn.BatchNorm2d(lays[2])
        self.fc1 = nn.Conv2d(lays[2]*4, 256, 1)
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

    def forward(self, x):
        # if self.orderpaths == True:

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        xm = self.pool3(x)
        xm = self.bn3_mag(xm)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm



class Net_steerinvariant_fmnist_scale(nn.Module):
    def __init__(self):
        super(Net_steerinvariant_fmnist_scale, self).__init__()

        lays = [12, 32, 48,60]
        kernel_sizes = [11, 11, 11,11]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        # Good configuration saved
        #
        # self.conv1 = ScaleConv_steering(1, 20, [kernel_sizes[0], kernel_sizes[0]], 1,
        #                                 padding=pads[0], sigma_phi_range=[np.pi/16],
        #                                  mode=1)
        # self.conv2 = ScaleConv_steering(20, 50, [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi/16],
        #                                 basis_scale = [0.2], mode=1)
        # self.conv3 = ScaleConv_steering(50, 100, [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi / 16],
        #                                 basis_scale = [0.1], mode=1)

        # For less data size
        lays = [30, 60, 90,1]

        # For 100 percent data size
        # lays = [30,60,90]
        self.conv0 = nn.Conv2d(1,12,3,padding=1)
        self.conv1 = ScaleConv_steering(12, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                        padding=pads[0], sigma_phi_range=[np.pi / 8],
                                        k_range=[0.5,1,2], ker_size_range=np.arange(7, 19, 2),
                                        # stride = 2,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        mode=1)
        self.conv2 = ScaleConv_steering(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
                                        k_range=[0.5,1,2], sigma_phi_range=[np.pi / 8],
                                        ker_size_range=np.arange(7, 19, 2),
                                        # stride=2,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi/16],
                                        mode=1, drop_rate=2)
        self.conv3 = ScaleConv_steering(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range=[0.5,1,2], sigma_phi_range=[np.pi / 8],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7, 19, 2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1, drop_rate=4)
        # self.conv4 = nn.Conv2d(lays[2],lays[3],3,padding=1)

        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(lays[0])
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(lays[1])
        self.pool3 = nn.MaxPool2d(8,padding=2)
        self.bn3 = nn.BatchNorm2d(lays[2])
        self.bn4 = nn.BatchNorm2d(lays[3])
        self.fc1 = nn.Conv2d(lays[2] * 4, 512, 1)
        self.fc1bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(512, 10, 1)  # FC2

    def forward(self, x):
        # if self.orderpaths == True:
        x = self.relu(self.conv0(x))
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        xm = self.pool3(x)
        xm = self.bn3(xm)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm


class Net_steerinvariant_cifar10_scale(nn.Module):
    def __init__(self):
        super(Net_steerinvariant_cifar10_scale, self).__init__()
        lnum = [24,48]

        # self.conv1 = ScaleConv_steering(3, lnum[0], [5,5], k_range = [0.5,1,2],
        #                                 ker_size_range=np.arange(3,9,2) )
        #
        # self.conv2 = ScaleConv_steering(lnum[0], lnum[0], [5,5], k_range = [0.5,1,2],
        #                                 ker_size_range=np.arange(3,9,2) )
        # self.conv3 = ScaleConv_steering(lnum[0], lnum[0], [5,5], k_range = [0.5,1,2],
        #                                 ker_size_range=np.arange(3,9,2),stride=2 )
        # self.conv4 = ScaleConv_steering(lnum[0], lnum[1], [5,5], k_range = [0.5,1,2],
        #                                 ker_size_range=np.arange(3,9,2) )
        # self.conv5 = ScaleConv_steering(lnum[1], lnum[1], [5,5], k_range = [0.5,1,2],
        #                                 ker_size_range=np.arange(3,9,2))
        # self.conv6 = ScaleConv_steering(lnum[1], lnum[1], [5,5], k_range = [0.5,1,2],
        #                                 ker_size_range=np.arange(3,9,2),stride=2 )
        # self.conv7 = nn.Conv2d(lnum[1], lnum[1], 3, padding=1)
        # self.conv8 = nn.Conv2d(lnum[1], lnum[1], 1)
        # self.class_conv = nn.Conv2d(lnum[1], 10, 1)


        #
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, 10, 1)



    def forward(self, x):

        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out



class Net_steerinvariant_stl10_scale(nn.Module):
    def __init__(self):
        super(Net_steerinvariant_stl10_scale, self).__init__()


        lays = [12, 32, 48]
        kernel_sizes = [11, 11, 11]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        # Good configuration saved
        #
        # self.conv1 = ScaleConv_steering(1, 20, [kernel_sizes[0], kernel_sizes[0]], 1,
        #                                 padding=pads[0], sigma_phi_range=[np.pi/16],
        #                                  mode=1)
        # self.conv2 = ScaleConv_steering(20, 50, [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi/16],
        #                                 basis_scale = [0.2], mode=1)
        # self.conv3 = ScaleConv_steering(50, 100, [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi / 16],
        #                                 basis_scale = [0.1], mode=1)

        # For less data size
        lays = [12,32,48,60,80]

        # For 100 percent data size
        # lays = [30,60,90]

        self.conv1 = ScaleConv_steering(3, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                        padding=pads[0], sigma_phi_range=[np.pi / 16],
                                        k_range = [0.5,1,2], ker_size_range=np.arange(7,19,2),
                                        # stride = 2,
                                        phi_range = np.linspace(0, np.pi, 9),
                                        phase_range = [-np.pi/4],
                                        mode=1)
        self.conv2 = ScaleConv_steering(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        ker_size_range=np.arange(7,19,2),
                                        # stride=2,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi/16],
                                        mode=1)
        self.conv3 = ScaleConv_steering(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7,19,2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1)

        self.conv4 = ScaleConv_steering(lays[2], lays[3], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range=[0.5, 1, 2], sigma_phi_range=[np.pi / 16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7, 19, 2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1)
        self.conv5 = ScaleConv_steering(lays[3], lays[4], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range=[0.5, 1, 2], sigma_phi_range=[np.pi / 16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7, 19, 2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1)


        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(lays[0])
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(lays[1])
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(lays[2])
        self.bn4 = nn.BatchNorm2d(lays[3])
        self.pool4 = nn.MaxPool2d(8)
        self.pool5 = nn.MaxPool2d(3)
        self.bn5 = nn.BatchNorm2d(lays[4])
        self.fc1 = nn.Conv2d(lays[3]*36, 512, 1)
        self.fc1bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(512, 10, 1)  # FC2

    def forward(self, x):
        # if self.orderpaths == True:

        x = self.conv1(x)
        # x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        # x = self.pool3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        xm = self.bn4(x)
        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm


