import random
import numpy as np
import sys
import gym
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.atari_wrappers import ClipRewardEnv,NoopResetEnv,EpisodicLifeEnv, FireResetEnv,StickyActionEnv, MaxAndSkipEnv
from domain.make_env import make_env
from neat_src import *

atari_env_names = ['Alien','Amidar','Assault','Asterix','Asteroids','Atlantis',\
                   'Berzerk','Bowling','Boxing','Breakout','Carnival','Centipede','Defender','Enduro',\
                   'Freeway','Frostbite','Gopher','Hero','Jamesbond','Kangaroo','Krull','Phoenix','Pitfall','Pong',\
                   'Pooyan','Qbert','Riverraid','Seaquest','Skiing','Solaris','Tennis','Tutankham','Venture','Zaxxon']
similar_atari_env_names = ['Assault','Beam_Rider','Demon_Attack','Space_Invaders','Phoenix']

single_atari_game = ['Phoenix']

class MyAtariWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = False,
    ) -> None:
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        #print(env.observation_space) #(128,)
        super().__init__(env)
        
class AdaptedMaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        if self._obs_buffer.shape != (2,128):
          self._obs_buffer = np.zeros((2,128),dtype=env.observation_space.dtype)
        self._skip = skip   
        
    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        #terminated = truncated = False
        for i in range(self._skip):
            obs, reward,done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info     


class AW_SB(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = AdaptedMaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        #print(env.observation_space) #(128,)
        super().__init__(env)

class ClipperWrapper(gym.RewardWrapper):
    def __init__(self,env: gym.Env) -> None:
        super().__init__(env)
    def reward(self, reward) -> float:
        return np.sign(float(reward))

class GymTask():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """
    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers      
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]
    self.ram      = game.ram
    self.full_actions = game.full
    # Environment
    self.nReps = nReps
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    if not paramOnly:
      self.env = make_env(game,render_mode=True,full_action=self.full_actions)
      #path= "/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/testMonitor"
      #self.env = Monitor(self.env,filename=path)
      self.env = AW_SB(self.env,clip_reward=False)

    
    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))   
  
  def getFitness(self, wVec, aVec, view=False, nRep=False, seed=-1):
    """Get fitness of a single individual.
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      nReps   - (nReps)    - number of trials to get average fitness
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = self.nReps
    wVec[np.isnan(wVec)] = 0
    reward = np.empty(nRep)
    for iRep in range(nRep):
      if seed > 0:
        seed = seed+iRep
      reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed)
    fitness = np.mean(reward)
    return fitness

  def testInd(self, wVec, aVec, hyp=None, view=False,seed=-1):
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    if seed >= 0:
      random.seed(seed)
      np.random.seed(seed)
      self.env.seed(seed)
    
    #next_game = str(np.random.choice(single_atari_game,1)[0])
    #self.env = make_env(next_game,render_mode=True)
    state = self.env.reset()
    self.env.t = 0
    #print("wVec",wVec.shape) #Atari (35,35) racing (20,20)
    #print("aVec",aVec.shape) #Atari (35,) racing (20,)
    #print("self.nInput",self.nInput) #Atari 16 racing 16
    #print("self.nOutput",self.nOutput) #Atari 18 racing 3
    #print("state",state.shape) #Atari (210,160,3) Racing (16,)
    annOut = act(wVec, aVec, self.nInput, self.nOutput, state)  
    action = selectAct(annOut,self.actSelect)
    #print("ACTION GENERATED BY SELECTACT IN TESTIND",action)
    #print("SHAPE OF ACTION GENERATED BY SELECTACT IN TESTIND",action.shape)   
    state, reward, done, info = self.env.step(action)
    
    if self.maxEpisodeLength == 0:
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      return reward
    else:
      totalReward = reward
    total_steps = 0
    for tStep in range(self.maxEpisodeLength): 
      total_steps +=1
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
      action = selectAct(annOut,self.actSelect) 
      state, reward, done, info = self.env.step(action)
      totalReward += reward  
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      if done:
        #print(total_steps)
        self.env.close()
        break
    return totalReward,int(total_steps)
