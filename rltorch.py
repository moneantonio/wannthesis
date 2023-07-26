import gym
import numpy as np
import argparse
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.atari_wrappers import ClipRewardEnv,NoopResetEnv,StickyActionEnv, MaxAndSkipEnv,EpisodicLifeEnv, FireResetEnv, AtariWrapper #TODO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import MlpExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import time
import os
import csv
BUDGET = 10*1e6
# python rltorch.py -o image -a 0 -g Boxing
# python rltorch.py -o ram -a 0 -g Boxing
#env = gym.make("CartPole-v1")
'''env = gym.make("Boxing-v0",obs_type="ram")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_boxing")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()
    if done:
      print("Reward",reward)
      obs = env.reset()'''

#how often checkpoint save? Test net every total_t iterations

column_names = ['highest','score','avg_score']

class AtariWrapperRAM(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = False,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        #print(env.observation_space) #(128,)
        super().__init__(env)
#original keeps clip_reward True

class TimeCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,starting,hours,filename,filepath, verbose=0):
        super(TimeCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.start_time = starting
        self.hours = hours
        self.filename = filename
        self.filepath = filepath

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        maxed = (60*60)*self.hours
        if (time.time()-self.start_time)>=(maxed):
            print("TIME OVER, TRAINING ENDED AFTER "+str(self.hours)+ " hours")
            self.model.save(self.filepath+'/'+self.filename+'_'+str(self.hours)+'h')
            self._on_training_end()
            exit()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class CNN2013(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CNN2015(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

net_arch_2013 = [256]

net_arch_2015 = [512]

class AtariRAMCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.observation_shape = observation_space.shape

        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1)

        self.fc_input_size = self._calculate_fc_input_size()

        self.fc = nn.Linear(self.fc_input_size, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #Flatten the tensor before passing it to the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def _calculate_fc_input_size(self):
        input = th.zeros(1, self.observation_shape[0])  # Create a dummy input
        x = input.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flattened_size = x.view(x.size(0), -1).size(1)
        return flattened_size
    

def train_agent(game,agent,hours,space,start,obs,seed=42,rep_buff=30000):
    '''first_check = False #hours
    second_check = False #hours*2
    third_check = False #hours*3
    fourth_check = False #hours*4
    fifth_check = False #hours*5
    sixth_check = False #hours*6
    seventh_check = False #hours*7
    eight_check = False #hours*8'''
    total_t = BUDGET
    device = ""
    entropy = 0.0
    valuef = 0
    parallels = 0
    steps=0
    batch=0
    
    #if th.backends.mps.is_available():
    #    device = 'mps'
    policy = ""
    obsname=""
    version = ""
    policy_kwargs = dict()
    ownpath = "/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/lastExperiment/aaa/"
    filename = ""
    #NETWORKS AND POLICIES
    if obs == 0:
        obsname = "ram"
        policy = "MlpPolicy"
        wrapper = AtariWrapperRAM
        device = 'cpu'
        if agent == 0:#DQN
            device = 'mps'
            policy_kwargs = dict(
                net_arch=net_arch_2015,
                optimizer_class = th.optim.RMSprop
            )
        if agent == 1:#PPO
            policy_kwargs = dict(
                net_arch=net_arch_2013,
                use_expln=True,
                #features_extractor_class=AtariRAMCNNFeatureExtractor,
                #features_extractor_kwargs=dict(),
                )
        elif agent == 2:#A2C
            policy_kwargs = dict(
                #net_arch=[256,128],
                net_arch=net_arch_2013,
                use_expln=True,
                optimizer_class = th.optim.RMSprop #already alpha 0.99
                )
        version = "-ram-v0"
        #version = "-ramNoFrameskip-v0"
    elif obs == 1:
        obsname = "image"
        policy = "CnnPolicy"
        wrapper = AtariWrapper
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu' 
        if agent == 0:#DQN
            device = 'mps'
            policy_kwargs = dict(
                net_arch=[]
            )
        if agent == 1:#PPO
            policy_kwargs = dict(
                net_arch=[],
                use_expln=True,
                features_extractor_kwargs=dict(),
                features_extractor_class=CNN2013
                )
        elif agent == 2:#A2C
            policy_kwargs = dict(
                net_arch=[],
                use_expln=True,
                features_extractor_class=CNN2013,
                features_extractor_kwargs=dict(),
                optimizer_class=RMSpropTFLike,
                optimizer_kwargs=dict(eps=1e-5)
            )
        version = "-v4"
    else:
        raise ValueError("Pick an observation type between ram and image")
    if agent == 0:#DQN
        name_agent = "DQN"
        filepath = game+'/'+name_agent+'/'+obsname
        filename = game+'_'+name_agent+'_'+obsname
        tmp_path = ownpath+filepath+'/'
        env = make_vec_env(env_id=game+version,seed=seed,n_envs=1,env_kwargs={"obs_type":obsname,"full_action_space":space},
                           monitor_dir= tmp_path,wrapper_class=wrapper)
        eval_env = VecMonitor(env, tmp_path)
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10,
                             deterministic=True, render=False)
        model = DQN(policy, env, verbose=1,learning_starts=50000,seed=seed,buffer_size=rep_buff,device=device,policy_kwargs=policy_kwargs)
    elif agent == 1:# PPO
        name_agent = "PPO"
        entropy = 0.01
        valuef = 1
        parallels= 8 #32
        steps = 128 
        batch = 32*parallels
        filepath = game+'/'+name_agent+'/'+obsname
        filename = game+'_'+name_agent+'_'+obsname
        tmp_path = ownpath+filepath+'/'
        env = make_vec_env(game+version,seed=seed, n_envs=parallels,
                           env_kwargs={"obs_type":obsname,"full_action_space":space},
                           monitor_dir = tmp_path,wrapper_class=wrapper)
        eval_env = VecMonitor(env, tmp_path)
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10,
                             deterministic=True, render=False)
        model = PPO(policy, env,ent_coef=entropy,vf_coef=valuef, verbose=1,seed=seed,device=device,policy_kwargs=policy_kwargs,
                    n_steps=steps,n_epochs=3,learning_rate=0.00025,batch_size=batch,clip_range=0.1
                    )
    elif agent == 2:#A2C
        name_agent = "A2C"
        entropy = 0.01 #0.1 for NTG
        valuef = 0.25
        parallels = 16 #32
        steps = 5 #256
        filepath = game+'/'+name_agent+'/'+obsname
        filename = game+'_'+name_agent+'_'+obsname
        tmp_path = ownpath+filepath+'/'
        env = make_vec_env(game+version, n_envs=parallels, env_kwargs={"obs_type":obsname,"full_action_space":space},
                           monitor_dir=tmp_path,wrapper_class=wrapper)
        #env = VecMonitor(env, tmp_path)
        eval_env = VecMonitor(env, tmp_path)
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10,
                             deterministic=True, render=False)
        model = A2C(policy,env,ent_coef=entropy,vf_coef=valuef,verbose=1,seed=seed,device=device,policy_kwargs=policy_kwargs,
                    learning_rate=0.001,gae_lambda=0.95,n_steps=steps,normalize_advantage=True
                    )
    print(model.policy)
    print("FILEPATH:", filepath)
    print("WRAPPER:",wrapper)
    print("Agent "+name_agent+" playing "+game+" with "+obsname+" representation")
    time_callback = TimeCallback(start,hours,filename,tmp_path)
    model.learn(total_timesteps=total_t, log_interval=10,callback=[time_callback,eval_callback],reset_num_timesteps=False)
    model.save(filepath)
    
    
def main(argv):
    start_time = time.time()
    #if argv.obs == 0:
    train_agent(argv.game,argv.agent,argv.time,argv.space,start_time,argv.obs,args.buff)
    #elif argv.obs == 1:
    #    test_agent_image(argv.game, argv.agent,argv.time,argv.space,start_time)
    
    

if __name__ == "__main__":
  ''' Parse input and launch '''
  
  parser = argparse.ArgumentParser(description=('Test SGDs'))
  
  parser.add_argument('-o', '--obs', type=int,\
   help='which observation to test, either 0 for ram or 1 for image', choices=[0, 1])
  parser.add_argument('-a','--agent',type=int,\
    help='which agent to test, 0 for DQN, 1 for PPO, 2 for A2C', choices=[0,1,2])
  parser.add_argument('-g','--game',type=str,\
    help='which game to play', default="Boxing")
  parser.add_argument('-t', '--time', type=int,\
   help='how much time do you want to train it for?', default=96)
  parser.add_argument('-b', '--buff', type=int,\
   help='size for the replay buffer of DQN, locally 30k otherwise it will not work due to memory limits', default=1000000)
  parser.add_argument('-s', '--space', type=bool,\
   help='full action space?', default=True)

  args = parser.parse_args()
  print(args)
  main(args)
