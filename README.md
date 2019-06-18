# SS-CNN
Scale-Steered CNN Implementation for PyTorch.
This is the official code repository for the paper 
"Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks" (https://arxiv.org/abs/1906.03861)

Please refer to ScaleSteerableInvariant_Network.py for the network library which contains the SS-CNN layer
along with various network architectures. 
The scale steerable basis functions are being created in scale_steering_lite.py. 
The dataset splits for MNIST-Scale, FMNIST-Scale and MNIST-Scale-local are provided as well. 

The main code where the SS-CNN is trained is in main_test.py. The dataset class is also included in that file,
where the images are being resized to twice their size and max-normalized (optional). 

To run on the different datasets, one can simply change the dataset_name paramter in the main function of main_test.py, 
along with the training size, batch size and total_epochs. 

Important to note is that the are predefined networks for each of the datasets in ScaleSteerableInvariant_Network.py, 
any of which can be used by changing the Networks_to_train parameter in main_test.py. 









