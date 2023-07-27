import numpy as np
import gym
from matplotlib.pyplot import imread

atari_env_names = ['Alien','Amidar','Assault','Asterix','Asteroids','Atlantis',\
                'Berzerk','Bowling','Boxing','Breakout','Carnival','Centipede','Defender','Enduro',\
                'Freeway','Frostbite','Gopher','Hero','Jamesbond','Kangaroo','Krull','Phoenix','Pitfall','Pong',\
                'Pooyan','Qbert','Riverraid','Seaquest','Skiing','Solaris','Tennis','Tutankham','Venture','Zaxxon']

def make_env(env_name, seed=-1,full_action = False, render_mode=False):
  # -- Bipedal Walker ------------------------------------------------ -- #
  if (env_name.env_name.startswith("BipedalWalker")):
    if (env_name.startswith("BipedalWalkerHardcore")):
      import Box2D
      from domain.bipedal_walker import BipedalWalkerHardcore
      env = BipedalWalkerHardcore()
    elif (env_name.startswith("BipedalWalkerMedium")): 
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()
      env.accel = 3
    else:
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()


  # -- VAE Racing ---------------------------------------------------- -- #
  elif (env_name.env_name.startswith("VAERacing")):
    from domain.vae_racing import VAERacing
    env = VAERacing()
    
  # -- Classification ------------------------------------------------ -- #
  elif (env_name.env_name.startswith("Classify")):
    from domain.classify_gym import ClassifyEnv
    if env_name.endswith("digits"):
      from domain.classify_gym import digit_raw
      trainSet, target  = digit_raw()
    
    if env_name.endswith("mnist784"):
      from domain.classify_gym import mnist_784
      trainSet, target  = mnist_784()
    
    if env_name.endswith("mnist256"):
      from domain.classify_gym import mnist_256
      trainSet, target  = mnist_256()

    env = ClassifyEnv(trainSet,target)  


  # -- Cart Pole Swing up -------------------------------------------- -- #
  elif (env_name.env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
    if (env_name.startswith("CartPoleSwingUp_Hard")):
      env.dt = 0.01
      env.t_limit = 200
      
  
  # -- ATARI ENV ------------------------------------------------------ --#
  else:
    from domain.atari_test import AtariTest
    from domain.atari_test_ram import AtariTestRam
    from domain.atari_raw import AtariRaw
    #next_game = np.random.choice(atari_env_names,1)
    #next_game = str(next_game[0])
    if env_name.ram == 0:
      env = AtariTestRam(gname=env_name, full_action=True)
    elif env_name.ram == 1:
      env = AtariTest(gname=env_name,full_action=True)#full_episode=True
    #print("NEW ENVIRONMENT ISTANTIATED WITH GAME",env.gname)
    elif env_name.ram == 2:
          env = AtariRaw(gname=env_name, full_action=full_action)

  if (seed >= 0):
    domain.seed(seed)

  return env