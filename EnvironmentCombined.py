from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.utils import shuffle
import gym
from gym import spaces
import os
from sklearn.metrics import accuracy_score
from skimage.transform import resize

def dice_score(gt, seg):
    
    k = 1
    dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
    
    return dice

class Environment(gym.Env):
    
    def __init__(self, x_train, task_predictor, img_shape, locations_train):
        
        self.x_train= x_train
        self.locations_train = locations_train        
        self.task_predictor = task_predictor
        
        
        self.controller_batch_size = 64
        self.task_predictor_batch_size = 32
        self.train_val_ratio = 0.6
        self.epochs_per_batch = 1
        
        self.crop_length = 28
        self.iter_tracker = 0
        
        self.train_batch_length = int(self.train_val_ratio * self.controller_batch_size)
        self.val_batch_length = int((1-self.train_val_ratio) * self.controller_batch_size)
        
        # (200, 200, 3, 2)
        self.img_shape = (200, 200, 3, 2)
                  
        self.observation_space =  spaces.Box(low=0, high=1, shape=self.img_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.current_state = 0       
        
        self.eval_rewards_hist = []
        self.eval_dice_hist = []
        self.actions_list = []
        
        self.episode_length = 256        
    
    def step(self, action):
        
        w = 40
        x = (np.clip(action[0] * 200, 0, 200 - w)).astype(int)
        y = (np.clip(action[1] * 200, 0, 200 - w)).astype(int)
        
        # Dice score calculation
        self.pred_img = self.current_image[x:x+w, y:y+w]                
        self.pred_img_for_dice = np.zeros((200, 200))
        self.gt_img_for_dice = np.zeros((200, 200))
        self.pred_img_for_dice[x:x+w, y:y+w] = 1
        self.gt_img_for_dice[self.current_location[0]:self.current_location[0]+self.current_location[2], self.current_location[1]:self.current_location[1]+self.current_location[3]] = 1        
        
        self.dice = dice_score(self.gt_img_for_dice, self.pred_img_for_dice)
        
        reward = self.dice

        # Image that is passed to classifier
        classifier_image = resize(self.pred_img, (200, 200, 3), preserve_range=True, anti_aliasing=False)        
        secondary_reward = self.task_predictor.predict(np.expand_dims(classifier_image, axis = 0))[0][0]
                
        self.iter_tracker += 1
        
        combined_images = np.concatenate([np.expand_dims(classifier_image, axis = -1), np.expand_dims(self.current_image, axis = -1)], axis = -1)
        next_observation = combined_images
        
        # Stop either at 80% classification score
        if secondary_reward > 0.8 or self.iter_tracker > self.episode_length:
            self.eval_rewards_hist.append(secondary_reward)
            self.eval_dice_hist.append(reward)
            
            done = True
            
        else:
            done = False
            
        return next_observation, secondary_reward, done, {}
        
    def reset(self):
        
        # Sample one image, crop randomly and pass that to controller
        current_idx = np.random.randint(0, len(self.x_train))
        
        self.current_image = self.x_train[current_idx,:,:,:]
        self.current_location = self.locations_train[current_idx].astype(int)
                
        # Taking random action
        random_action = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32).sample()
        
        w = 40
        x = (np.clip(random_action[0] * 200, 0, 200 - w)).astype(int)
        y = (np.clip(random_action[1] * 200, 0, 200 - w)).astype(int)
        
        # Cropped image
        image = resize(self.current_image[x:x+w, y:y+w], (200, 200, 3), preserve_range=True, anti_aliasing=False)
        image = np.expand_dims(image, axis = -1)
        
        self.iter_tracker = 0
        
        combined_images = np.concatenate([np.expand_dims(self.current_image, axis = -1), image], axis = -1)
            
        return combined_images