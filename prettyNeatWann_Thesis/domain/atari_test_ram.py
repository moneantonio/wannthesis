import numpy as np
import gym
import time

#from scipy.misc import imresize as resize
from scipy import misc
from skimage.transform import resize
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras import backend as B 
#from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, GaussianNoise, Lambda, InputLayer
from gym.spaces.box import Box
from vae.vae import ConvVAE
from gym.envs.atari import AtariEnv
import tensorflow as tf
import tensorflow.keras as keras
import sys
from domain.config import games
#sys.path.append('domain/')
#from config import games

#current_game = games['atari_stack16']
#current_game = games['atari_stack32']
#current_game = games['atari_stack128']
#current_game = games['atari_stack192']
#current_game = games['atari_stack256']
#current_game = games['atari_stack512']
#current_game = games['atari_stack1024']

#from expandedvae import MyVAE
import json
#from complexvae import MyVariationalAutoencoder
#from complexvae import ComplexVAE

MU_MODE = False

atari_env_names = ['Alien','Amidar','Assault','Asterix','Asteroids','Atlantis',\
                   'Berzerk','Bowling','Boxing','Breakout','Carnival','Centipede','Defender','Enduro',\
                   'Freeway','Frostbite','Gopher','Hero','Jamesbond','Kangaroo','Krull','Phoenix','Pitfall','Pong',\
                   'Pooyan','Qbert','Riverraid','Seaquest','Skiing','Solaris','Tennis','Tutankham','Venture','Zaxxon']

SCREEN_X = 210
SCREEN_Y = 160
SCREEN = 160
#latent_dim = 128
TIME_LIMIT = 1500
def _clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)
#this needs to be resized
def _process_frame(frame):
  obs = frame[:, :, :].astype(np.float)/255.0
  obs = resize(obs, (SCREEN, SCREEN))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

def _process_frame_green(frame):
  obs = frame[:, :, :].astype(np.float)/255.0
  obs = resize(obs, (210, 160))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs[:, :, 1] # green channel

class AtariTestRam(AtariEnv):
    def __init__(self,gname, full_episode=False,full_action = True):#, gname = current_game):
        self.gname = gname.env_name#+"NoFrameskip-v4"
        super(AtariTestRam, self).__init__(game=self.gname,obs_type="ram",full_action_space=full_action)
        #print("GNAME IN CLASS:",self.gname,"LEN OF POSSIBLE ACTIONS",len(self.unwrapped.get_action_meanings()),"POSSIBLE ACTIONS",self.unwrapped.get_action_meanings())
        #self.env = gym.make("Assault-v0")
        self._internal_counter = 0
        self.z_size = gname.input_size
        #print(self.z_size)
            #self.vae = ConvVAE(batch_size=1, z_size=self.z_size, gpu_mode=False, is_training=False, reuse=True)
        #self.vae.load_json('vae/vae_'+str(self.z_size)+'.json')
        #self.vae.load_json('vae/vae_16.json')
        #self.vae = ComplexVAE
        #path = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWann/models/'
        #path = '/home/s3113159/prettyNeatWann/models/'
        #correctmodel = self.gname+'/z'+str(self.z_size)+'/'+str(SCREEN)+'_'+str(SCREEN)+'/encoder'
        #pathenc = path+correctmodel
        #self.vae = keras.models.load_model(pathenc,compile=False)
        self.full_episode = full_episode
        high = np.array([np.inf] * self.z_size)
        #self.observation_space = (210,160,3)
        #print("ATARI SELF OBSERVATION SPACE",self.observation_space.shape)
        self._has_rendered = False
        self.real_frame = None

    def reset(self):
        #print("ENTER ATARI RESET")
        self._internal_counter = 0
        self._has_rendered = False
        self.real_frame = None
        #print("ATARI RESET SHAPE",(super(AtariTest,self).reset()).shape) #(210,160,3)
        #state = np.resize(super(AtariTest, self).reset(),(self.z_size,))
        
        '''newscreen = np.copy(_process_frame(super(AtariTest, self).reset())).astype(np.float)/255.0
        newscreen = newscreen.reshape(1, SCREEN, SCREEN, 3)
        state,_,_ = self.vae(newscreen)'''
        state = np.copy(super(AtariTestRam,self).reset()).astype(np.float)/255.0
        #state = super(AtariTest,self).reset()
        return state

    def render(self, mode='human', close=False):
        if mode == 'human' or mode == 'rgb_array':
            self._has_rendered = True
        return super(AtariTestRam, self).render(mode=mode)
    
    def seed(self,seed = None):
        return super(AtariTestRam,self).seed(seed=seed)

    def step(self, action):
        
        #print("ENTER ATARI STEP")
        if not self._has_rendered:
            self.render("rgb_array")
            self._has_rendered = False
            
        action = np.argmax(action)
        #print("ENVIRONMENT POSSIBLE ACTIONS", self.env.action_space)
        #print("ACTIONS IN STEP:",action)
        
        obs, reward, done, _ = super(AtariTestRam,self).step(action)
        #print("USING RAM")
        #print("OBS SHAPE:",obs.shape)
        
        '''result = np.copy(_process_frame(obs)).astype(np.float)/255.0'''
        z = np.copy(obs).astype(np.float)/255.0
        
        #print("RESULTS SHAPE (OBS AFTER PROCESS FRAME FUNCTION):",result.shape) #(210,160,3)
        #result = resize(result,(128,128,3))
        '''result = result.reshape(1, SCREEN, SCREEN, 3)
        self.real_frame = result'''

        #z = self.vae.encode(result).flatten()
            #mu, logvar = self.vae.encode_mu_logvar(result)
        #print("VAE OPERATION : mu -> SHAPE",mu.shape)
        #print("VAE OPERATION : logvar -> SHAPE",logvar.shape)
            #mu = mu[0]
            #logvar = logvar[0]
        #print("VAE OPERATION : mu[0] -> SHAPE",mu.shape)
        #print("VAE OPERATION : logvar[0] -> SHAPE",logvar.shape)
            #s = logvar.shape
            #z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
           
        '''z,mu,logvar = self.vae(result)
        s = logvar.shape
        z = mu + np.exp(logvar/2.0) * np.random.randn(*s)'''
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
            super(AtariTestRam, self).close()

        if MU_MODE:
            return mu, reward, done, {}
        return z, reward, done, {}
