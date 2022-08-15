
import os
import random
import shutil
import numpy as np
import torch


class TransWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img):
        return self.seq.augment_image(img)


class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)

max_imu = np.array([
    0.0012899759199437844,
    0.004137401473796877,
    0.9999999640335174,
    0.9999999825684567,
    0.2392873764038086,
    0.0510830394923687,
    0.813437819480896,
    16.141658782958984,
    20.80577850341797,
    11.635208129882812
])

min_imu = np.array([
    -0.004909195489332213,
    -0.004784990981824879,
    -0.9999999946240098,
    3.172563338587483e-05,
    -0.273921400308609,
    -0.09309031069278717,
    -1.6021742820739746,
    -27.349824905395508,
    -22.472227096557617,
    5.817458629608154
])
# max_imu = np.array([ 9.8419981,10.21207809,10.56573486, 10.91194344, 11.23107529, 11.55004311, 11.885849,   12.21730995, 12.54627323, 12.87323761])
# min_imu = np.array([-1.69137157e-02, -5.93055412e-03, -3.24095320e-03, -9.20971972e-04, -5.17153597e+00, -1.02396202e+01, -1.48390427e-01, -7.01377913e-02, -3.38135324e-02, -1.66577529e-02])

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
  
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def normalize_imu(imu):
    global max_imu, min_imu
    imu = (imu-min_imu)/(max_imu-min_imu)
    return imu


def normalize_speed(speed):
#     maxi=12.177567
#     mini=-5.12704
    maxi = 25.453960418701172
    mini = -0.0002273327118018642
    speed=(speed-mini)/(maxi-mini)
    return speed


def normalize_steering(steering):
    steering=(steering+1.0)/2.0
    return steering


def normalize_image(image):
    return image/255.0


def save_checkpoint(state, id_, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename,
            os.path.join("save_models", "{}_best.pth".format(id_))
            )