import numpy as np
import gym

#from scipy.misc import imresize as resize
from scipy import misc
from skimage.transform import resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

from vae.vae import ConvVAE
import sys
sys.path.append('domain/')
from config import games

import json

#import imageio

SCREEN_X = 64
SCREEN_Y = 64

TIME_LIMIT = 1000

MU_MODE = True

#all x between 0.0 and 1.0
def _clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  #print("OBS SHAPE IN THE PROCESS FRAME",obs.shape)
  obs = resize(obs, (64, 64))
  #print("OBS SHAPE IN PROCESS FRAME AFTER RESIZE",obs.shape)
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  #print("OBS SHAPE IN PROCESS FRAME AFTER LAST OPERATION BEFORE RETURN",obs.shape)
  return obs

def _process_frame_green(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs[:, :, 1] # green channel

class VAERacing(CarRacing):
  def __init__(self, full_episode=False):
    super(VAERacing, self).__init__()
    self._internal_counter = 0
    self.z_size = games['vae_racing'].input_size
    self.vae = ConvVAE(batch_size=1, z_size=self.z_size, gpu_mode=False, is_training=False, reuse=True)
    self.vae.load_json('vae/vae_'+str(self.z_size)+'.json')
    self.full_episode = full_episode
    high = np.array([np.inf] * self.z_size)
    self.observation_space = Box(-high, high)
    #print("VAERACING OBSERVATION SPACE SHAPE",self.observation_space.shape) #(16,)
    self._has_rendered = False
    self.real_frame = None

  def reset(self):
    self._internal_counter = 0
    self._has_rendered = False
    self.real_frame = None
    #print("RACING RESET SHAPE",(super(VAERacing,self).reset()).shape) #(16,)
    return super(VAERacing, self).reset()

  def render(self, mode='human', close=False):
    if mode == 'human' or mode == 'rgb_array':
      self._has_rendered = True
    return super(VAERacing, self).render(mode=mode)

  def step(self, action):

    if not self._has_rendered:
      self.render("rgb_array")
      self._has_rendered = False

    if action is not None:
      action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
      action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
      action[1] = (action[1]+1.0) / 2.0
      action[2] = _clip(action[2])
    #print("ACTIONS IN STEP:",action)
    obs, reward, done, _ = super(VAERacing, self).step(action)
    #print("OBS SHAPE:",obs.shape) #(96,96,3)
    #process_frame mi fa una normalizzazione dello schermo nel nuovo stato
    result = np.copy(_process_frame(obs)).astype(np.float)/255.0
    #print("RESULTS SHAPE (OBS AFTER PROCESS FRAME FUNCTION):",result.shape) #(64,64,3)

    result = result.reshape(1, 64, 64, 3)
    self.real_frame = result

    #z = self.vae.encode(result).flatten()
    mu, logvar = self.vae.encode_mu_logvar(result)
    #print("VAE OPERATION : mu -> SHAPE",mu.shape)
    #print("VAE OPERATION : logvar -> SHAPE",logvar.shape)
    mu = mu[0]  
    logvar = logvar[0]
    #print("VAE OPERATION : mu[0] -> SHAPE",mu.shape)
    #print("VAE OPERATION : logvar[0] -> SHAPE",logvar.shape)
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    #print("FINAL VAE OPERATION: Z LATENT VECTOR -> ", z)
    #print("FINAL VAE OPERATION: Z.SHAPE LATENT VECTOR -> ", z.shape)

    if self.full_episode:
      if MU_MODE:
        return mu, reward, False, {}
      else:
        return z, reward, False, {}

    self._internal_counter += 1
    if self._internal_counter > TIME_LIMIT:
      done = True

    if MU_MODE:
      return mu, reward, done, {}
    return z, reward, done, {}
