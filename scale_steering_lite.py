import numpy as np
import scipy.ndimage


def generate_filter_basis(filter_size,phi0,sigma,k,scale,phase,drop_rate):

    # delta_r = 0.15
    rot_k=0
    Mx = (filter_size[0])
    My = (filter_size[1])
    W = np.ones((filter_size[0], filter_size[1]))
    W[np.int((Mx-1)/2), np.int((My-1)/2)] = 0
    W_dist = scipy.ndimage.morphology.distance_transform_bf(W)

    W_dist[np.int((Mx - 1) / 2), np.int((My - 1) / 2)] = 0
    Mask = np.ones(W_dist.shape)
    Mask[W_dist > np.int((Mx - 1) / 2)] = 0
    W_dist[np.int((Mx - 1) / 2), np.int((My - 1) / 2)] = 1
    W_dist = scale*W_dist
    # W_dist = W_dist + delta_r
    log_r = np.log(W_dist)
    # log_r[np.int((Mx-1)/2), np.int((My-1)/2)] = 1

    x_coords = np.zeros((filter_size[0], filter_size[1]))
    y_coords = np.zeros((filter_size[0], filter_size[1]))

    for i in range(x_coords.shape[0]):
        x_coords[i,:] = (((Mx-1)/2) - i)

    for i in range(y_coords.shape[1]):
        y_coords[:, i] = -(((My-1)/2) - i)


    phi_image = scipy.arctan2(y_coords,x_coords)
    L1 = np.abs(np.minimum(np.abs(phi_image-phi0),np.abs(phi_image+2*np.pi-phi0)))
    L2 = np.abs(np.minimum(np.abs(phi_image - phi0-np.pi), np.abs(phi_image + 2 * np.pi - phi0-np.pi)))
    # L2 = np.minimum(L1,np.abs(phi_image+np.pi-phi0))
    exp_phi = np.exp(-np.power(np.minimum(L2,L1),2.0)/(2*sigma*sigma))*(1.0/W_dist)
    # exp_phi = (1.0/(np.power(L1/sigma,1)+1))*(1.0/W_dist)
    # plt.imshow(exp_phi)
    # plt.pause(0.2)
    # plt.draw()
    # print(np.min(log_r)/np.pi,np.max(log_r)/np.pi)

    effective_k = 2*np.pi*k/np.log(np.max(W_dist))
    # C = np.minimum(L2,L1)
    # print(effective_k)
    filter_real = exp_phi*np.cos((effective_k*(log_r))+phase)*Mask
    filter_imag = exp_phi*np.sin((effective_k*(log_r))+phase)*Mask

    # filter_real[np.int((Mx-1)/2), np.int((My-1)/2)] = 0
    # filter_imag[np.int((Mx-1)/2), np.int((My-1)/2)] = 0
    # filter_real = filter_real/np.linalg.norm(exp_phi)
    # filter_imag = filter_imag/np.linalg.norm(exp_phi)

    # Final step: Normalize Energies

    return filter_real,filter_imag, effective_k