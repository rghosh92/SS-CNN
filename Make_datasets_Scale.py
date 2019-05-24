import torchvision
import numpy as np
import os
import random
import torchvision.transforms as transforms
from scipy.ndimage import zoom
import pickle

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))

        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding

        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2


        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        #  trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)


        if trim_top<0:
            out = img
            print(zoom_factor)

        else:
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]


    # If zoom_factor == 1, just return the input array
    else:
        out = img


    return out


def make_stl10_scale():


    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])

    trainset = torchvision.datasets.STL10(root='./data',split='train',
                                          download=True, transform=transform)

    testset = torchvision.datasets.STL10(root='./data',split='test',
                                         download=True, transform=transform)

    train_data = trainset.data/255.0
    test_data = testset.data/255.0
    train_label = trainset.labels
    test_label = testset.labels

    try:
        os.mkdir('STL10/')
    except:
        None

    os.chdir('STL10/')

    dict = {}
    dict['train_data'] = train_data
    dict['train_label'] = train_label

    dict['test_data'] = test_data
    dict['test_label'] = test_label

    pickle.dump(dict, open('STL10.pickle', 'wb'))

    os.chdir("..")






def make_only_mnist():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)


    all_data = np.zeros([70000,28,28])
    all_data[0:60000,:,:] = trainset.data.numpy()/255.0

    all_data[60000:70000,:,:] = testset.data.numpy()/255.0
    all_targets = np.zeros(70000)
    all_targets[0:60000] = trainset.targets.numpy()
    all_targets[60000:70000] = testset.targets.numpy()

    train_data = all_data[0:60000,:,:]
    train_label = all_targets[0:60000]
    test_data = all_data[60000:70000,:,:]
    test_label = all_targets[60000:70000]

    print(np.min(all_data))
    print(np.max(all_data))

    try:
        os.mkdir('MNIST/')
    except:
        None

    os.chdir('MNIST/')


    dict = {}
    dict['train_data']  = train_data
    dict['train_label']  = train_label

    dict['test_data'] = test_data
    dict['test_label'] = test_label

    pickle.dump(dict,open('mnist_only_split.pickle','wb'))

    os.chdir("..")

def make_mnistlocal_scale(val_splits):


    os.chdir('MNIST_SCALE_LOCAL/')

    dict = pickle.load(open('pad_two_scale_image_522.pickle','rb'))
    all_data = dict['data']
    all_targets = dict['label']

    for split in range(val_splits):
        train_data = np.zeros([10000,28,39])
        train_label = np.zeros([10000])
        test_data = np.zeros([40000,28,39])
        test_label = np.zeros([40000])


        random.seed(split)
        perm = np.random.permutation(all_data.shape[0])
        all_data = all_data[perm,:,:]
        all_targets = all_targets[perm]

        i = 0
        for j in range(10000):
            train_data[j,:,:] = all_data[i,:,:]
            train_label[j] = all_targets[i]
            i += 1



        for j in range(40000):
            test_data[j,:,:] = all_data[i,:,:]
            test_label[j] = all_targets[i]
            i += 1



        dict = {}
        dict['train_data']  = train_data
        dict['train_label']  = train_label


        dict['test_data'] = test_data
        dict['test_label'] = test_label

        pickle.dump(dict,open('mnist_scale_local2_split_' + str(split) + '.pickle','wb'))

    os.chdir("..")


def make_mnist_scale(val_splits):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)


    all_data = np.zeros([70000,28,28])
    all_data[0:60000,:,:] = trainset.data.numpy()/255.0
    all_data[60000:70000,:,:] = testset.data.numpy()/255.0
    all_targets = np.zeros(70000)
    all_targets[0:60000] = trainset.targets.numpy()
    all_targets[60000:70000] = testset.targets.numpy()

    print(np.min(all_data))
    print(np.max(all_data))

    try:
        os.mkdir('MNIST_SCALE/')
    except:
        None

    os.chdir('MNIST_SCALE/')

    for split in range(val_splits):
        train_data = np.zeros([10000,28, 28 ])
        train_label = np.zeros([10000])
        train_scale = np.zeros([10000])
        val_data = np.zeros([2000,28, 28 ])
        val_label = np.zeros([2000])
        val_scale = np.zeros([2000])
        test_data = np.zeros([50000,28, 28 ])
        test_label = np.zeros([50000])
        test_scale = np.zeros([50000])


        random.seed(split)
        perm = np.random.permutation(all_data.shape[0])
        all_data = all_data[perm,:,:]
        all_targets = all_targets[perm]

        i = 0
        for j in range(10000):
            zoom_factor = 0.3 + (np.random.rand()*0.7)
            train_data[j,:,:] = clipped_zoom(all_data[i,:,:],zoom_factor,order=1)
            train_label[j] = all_targets[i]
            train_scale[j] = zoom_factor
            i += 1

        for j in range(2000):
            zoom_factor = 0.3 + (np.random.rand() * 0.7)
            val_data[j,:,:] = clipped_zoom(all_data[i,:,:], zoom_factor, order=1)
            val_label[j] = all_targets[i]
            val_scale[j] = zoom_factor
            i += 1


        for j in range(50000):
            zoom_factor = 0.3 + (np.random.rand() * 0.7)
            test_data[j,:,:] = clipped_zoom(all_data[i,:,:], zoom_factor, order=1)
            test_label[j] = all_targets[i]
            test_scale[j] = zoom_factor
            i += 1



        dict = {}
        dict['train_data']  = train_data
        dict['train_label']  = train_label
        dict['train_scale']  = train_scale

        dict['val_data'] = val_data
        dict['val_label'] = val_label
        dict['val_scale'] = val_scale

        dict['test_data'] = test_data
        dict['test_label'] = test_label
        dict['test_scale'] = test_scale

        pickle.dump(dict,open('mnist_scale_split_' + str(split) + '.pickle','wb'))

    os.chdir("..")


def make_fmnist_scale(val_splits):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                          download=True, transform=transform)


    all_data = np.zeros([70000,28,28])
    all_data[0:60000,:,:] = trainset.data.numpy()/255.0
    all_data[60000:70000,:,:] = testset.data.numpy()/255.0
    all_targets = np.zeros(70000)
    all_targets[0:60000] = trainset.targets.numpy()
    all_targets[60000:70000] = testset.targets.numpy()

    print(np.min(all_data))
    print(np.max(all_data))

    try:
        os.mkdir('FMNIST_SCALE_NEW/')
    except:
        None

    os.chdir('FMNIST_SCALE_NEW/')

    for split in range(val_splits):
        train_data = np.zeros([10000,28, 28 ])
        train_label = np.zeros([10000])
        train_scale = np.zeros([10000])
        val_data = np.zeros([2000,28, 28 ])
        val_label = np.zeros([2000])
        val_scale = np.zeros([2000])
        test_data = np.zeros([50000,28, 28 ])
        test_label = np.zeros([50000])
        test_scale = np.zeros([50000])


        random.seed(split)
        perm = np.random.permutation(all_data.shape[0])
        all_data = all_data[perm, :, :]
        all_targets = all_targets[perm]

        i = 0
        for j in range(10000):
            zoom_factor = 0.7 + (np.random.rand()*0.3)
            train_data[j,:,:] = clipped_zoom(all_data[i,:,:],zoom_factor,order=1)
            train_label[j] = all_targets[i]
            train_scale[j] = zoom_factor
            i += 1

        for j in range(2000):
            zoom_factor = 0.7 + (np.random.rand() * 0.3)
            val_data[j,:,:] = clipped_zoom(all_data[i,:,:], zoom_factor, order=1)
            val_label[j] = all_targets[i]
            val_scale[j] = zoom_factor
            i += 1


        for j in range(50000):
            zoom_factor = 0.7 + (np.random.rand() * 0.3)
            test_data[j,:,:] = clipped_zoom(all_data[i,:,:], zoom_factor, order=1)
            test_label[j] = all_targets[i]
            test_scale[j] = zoom_factor
            i += 1



        dict = {}
        dict['train_data']  = train_data
        dict['train_label']  = train_label
        dict['train_scale']  = train_scale

        dict['val_data'] = val_data
        dict['val_label'] = val_label
        dict['val_scale'] = val_scale

        dict['test_data'] = test_data
        dict['test_label'] = test_label
        dict['test_scale'] = test_scale

        pickle.dump(dict,open('Fmnist_scale_split_' + str(split) + '.pickle','wb'))

    os.chdir("..")



def make_cifar10_scale(val_splits):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)


    # all_data_train = np.zeros([60000,32,32,3])
    all_data_train = trainset.data/255.0
    all_targets_train = trainset.targets

    # all_data_test = np.zeros([10000,32,32,3])
    all_data_test = testset.data/255.0
    all_targets_test = testset.targets

    try:
        os.mkdir('CIFAR10_SCALE/')
    except:
        None

    os.chdir('CIFAR10_SCALE/')

    for split in range(val_splits):
        train_data = np.zeros([50000,32,32,3 ])
        train_label = np.zeros([50000])
        # train_scale = np.zeros([10000])
        # val_data = np.zeros([2000,28, 28 ])
        # val_label = np.zeros([2000])
        # val_scale = np.zeros([2000])
        test_data = np.zeros([30000,32,32,3])
        test_label = np.zeros([30000])
        test_scale = np.zeros([30000])


        # random.seed(split)
        # np.random.shuffle(all_data)

        i = 0
        for j in range(50000):
            train_data[j,:,:,:] = all_data_train[i,:,:,:]
            train_label[j] = all_targets_train[i]
            i += 1

        j = 0
        for i in range(10000):
            test_data[j,:,:,:] = all_data_test[i,:,:,:]
            test_label[j] = all_targets_test[i]
            test_scale[j] = 1
            j += 1
            test_data[j,:,:,:] = clipped_zoom(all_data_test[i,:,:,:], zoom_factor = 32.0/28.0 , order=3)
            test_label[j] = all_targets_test[i]
            test_scale[j] = 32.0/28.0
            j += 1
            test_data[j,:,:,:] = clipped_zoom(all_data_test[i,:,:,:], zoom_factor = 32.0/24.0 , order=3)
            test_label[j] = all_targets_test[i]
            test_scale[j] = 32.0 / 24.0
            j += 1



        dict = {}
        dict['train_data']  = train_data
        dict['train_label']  = train_label
        # dict['train_scale']  = train_scale

        # dict['val_data'] = val_data
        # dict['val_label'] = val_label
        # dict['val_scale'] = val_scale

        dict['test_data'] = test_data
        dict['test_label'] = test_label
        dict['test_scale'] = test_scale

        pickle.dump(dict,open('cifar10_scale_split_' + str(split) + '.pickle','wb'))

    os.chdir("..")




if __name__ == "__main__":
    make_mnistlocal_scale(6)
    # make_mnist_scale(6)
    # make_fmnist_scale(6)
    # make_cifar10_scale(1)
    # make_only_mnist()
    # make_stl10_scale()