import os
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3
from stable_baselines3.common.env_checker import check_env

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import random
from sklearn.utils import shuffle
import os
from sklearn.metrics import accuracy_score
from skimage.transform import resize
import tensorflow.keras.backend as K
from EnvironmentCombined import Environment
from sklearn.model_selection import train_test_split



# Loading datasets
images_path = 'IMAGES_WITH.npy'
locations_path = 'LOCATIONS.npy'

images = np.load(images_path).T
locations = np.load(locations_path)

# Normalising images
def normaliser(image_arr):
    
    img_min = np.min(image_arr)
    img_max = np.max(image_arr - img_min)
    
    image_arr = (image_arr - img_min) / img_max
    
    return np.expand_dims(image_arr, axis = -1)

images_1 = normaliser(images[:,:,:,0])
images_2 = normaliser(images[:,:,:,1])
images_3 = normaliser(images[:,:,:,2])

IMAGES = np.concatenate([images_1, images_2], axis = -1)
IMAGES = np.concatenate([IMAGES, images_3], axis = -1)

print('read')

# Preparing for training
input_shape = IMAGES[1,:,:,:].shape

X = IMAGES
Y = locations

predictor = tf.keras.models.load_model('predictor')

def make_env():
    #return Environment(X, Y, X_val, Y_val, predictor, input_shape, LOCATIONS_TRAIN, LOCATIONS_TEST)
    return Environment(X, predictor, input_shape, Y)

n_envs = 1
env = make_vec_env(make_env, n_envs = n_envs)
ep_length = env.get_attr('episode_length', 0)[0]

# 1/4 as last resort
batch_size = 1*ep_length #*n_envs

model = PPO("MlpPolicy", env, verbose=1, n_steps=1*ep_length, batch_size=batch_size)

num_ep_per_trial = 64

num_trials = 8*1024

num_ep_eval = 2

length = 0
i = 0

while length < 250:
    
    model.learn(total_timesteps = num_ep_per_trial*ep_length)
    
    eval_rewards_hist = env.get_attr('eval_rewards_hist', 0)[0]
    eval_dice_hist = env.get_attr('eval_dice_hist', 0)[0]
    
    length = len(eval_dice_hist)
    
    print('Mean Dice Score: ' + str(np.mean(eval_dice_hist)))
    print('Length: ' + str(length))
    print('Iteration: ' +  str(i))
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1)
    ax[0].plot(eval_rewards_hist)
    ax[1].plot(eval_dice_hist)
    fig.savefig('reward_current.png')
    
    i += 1

print(np.mean(eval_dice_hist))
model.save("model_current_" + str(i))