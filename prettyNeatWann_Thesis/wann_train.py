import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
np.set_printoptions(precision=2, linewidth=160) 

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

MILLION = int(1e6)
BUDGET = 10*MILLION
# prettyNeat
from neat_src import * # NEAT and WANNs
from domain import *   # Task environments

games_dict = {
  "Breakout": 4,
  "Pong": 6,
  "Boxing": 18,
  #"Video_Pinball": 9,
  "Phoenix": 18,
  "Qbert": 18,
  "Battle_Zone":18,
  "Name_This_Game":18,
  "Double_Dunk":18,
  "Centipede" : 18,
  "Robotank" : 18,
  "Tennis" : 18,
  "Ice_Hockey" : 18,
  "Up_N_Down" : 6,
  "Carnival" : 6,
  "Atlantis" : 4
}


# -- Run NEAT ------------------------------------------------------------ -- #
def master(starting_time_total,hours): 
  """Main NEAT optimization script
  """
  
  global fileName, hyp
  steps = BUDGET
  print("START TRAINING ------- > filename "+fileName+" evolving for a maximum of "+str(hours)+" hours")
  print(f"TOTAL BUDGET OF {steps} TIMESTEPS")
  data = WannDataGatherer(fileName, hyp)
  alg  = Wann(hyp)

  for gen in range(hyp['maxGen']):
    gen_start_time = time.time()      
    pop = alg.ask()            # Get newly evolved individuals from NEAT
    print("TIME FOR CREATION OF NEW POPULATION  ",time.time()-gen_start_time, "seconds")
    #print("size of pop",sys.getsizeof(pop)/1e06,'MB')
    reward,steps = batchMpiEval(pop,steps)  # Send pop to be evaluated by workers
    print("Reward",reward, "\n Generation completed in --- %s seconds ---" % (time.time() - gen_start_time))#
    total_elapsed_time= (time.time()-starting_time_total)/60
    print(f"Remaining Budget: {steps} steps ---- total time elapsed: {total_elapsed_time} minutes")
    alg.tell(reward)           # Send fitness to NEAT    

    data = gatherData(data,alg,gen,hyp)
    print(gen, '\t', data.display())
    '''if (time.time()-starting_time_total)>=(900):
          print("TIME OVER, EVOLUTION ENDED AFTER "+str(hours)+ " hours in which a total of "+str(gen+1)+" generations have been evolved in "+str(BUDGET)+"steps")
          break'''
    if (time.time()-starting_time_total)>=((60*60)*hours):
          print("TIME OVER, EVOLUTION ENDED AFTER "+str(hours)+ " hours in which a total of "+str(gen+1)+" generations have been evolved in "+str(BUDGET)+"steps")
          break
    if steps < 0 :
          print("BUDGET OF "+str(BUDGET)+" STEPS IS OVER, EVOLUTION ENDED AFTER "+str((time.time()-starting_time_total)/(60*60))+ " hours in which a total of "+str(gen+1)+" generations have been evolved")
          break

  # Clean up and data gathering at run end
  data = gatherData(data,alg,gen,hyp,savePop=True)
  data.save()
  data.savePop(alg.pop,fileName) # Save population as 2D numpy arrays
  stopAllWorkers()

def gatherData(data,alg,gen,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    alg        - (Wann)          - neat algorithm container
      .pop     - [Ind]           - list of individuals in population    
      .species - (Species)       - current species
    gen        - (ind)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(alg.pop, alg.species)
  if (gen%hyp['save_mod']) == 0:
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks    
    global fileName
    pref = 'log/' + fileName
    import pickle
    with open(pref+'_pop.obj', 'wb') as fp:
      pickle.dump(alg.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = max(hyp['bestReps'], (nWorker-1))
    rep = np.tile(data.best[-1], bestReps)
    fitVector,steps = batchMpiEval(rep, sameSeedForEachIndividual=False)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!      
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data


# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(pop,steps=0, sameSeedForEachIndividual=True):
  """Sends population to workers for evaluation one batch at a time.

  Args:
    pop - [Ind] - list of individuals
      .wMat - (np_array) - weight matrix of network
              [N X N] 
      .aVec - (np_array) - activation function of each node
              [N X 1]

  Return:
    reward  - (np_array) - fitness value of each individual
              [N X 1]

  Todo:
    * Asynchronous evaluation instead of batches
  """
  global nWorker, hyp
  nSlave = nWorker-1
  nJobs = len(pop)
  nBatch= math.ceil(nJobs/nSlave) # First worker is master

    # Set same seed for each individual
  if sameSeedForEachIndividual is False:
    seed = np.random.randint(1000, size=nJobs)
  else:
    seed = np.random.randint(1000)

  reward = np.empty( (nJobs,hyp['alg_nVals']), dtype=np.float64)
  i = 0 # Index of fitness we are filling
  for iBatch in range(nBatch): # Send one batch of individuals
    for iWork in range(nSlave): # (one to each worker if there)
      if i < nJobs:
        wVec   = pop[i].wMat.flatten()
        n_wVec = np.shape(wVec)[0]
        aVec   = pop[i].aVec.flatten()
        n_aVec = np.shape(aVec)[0]
        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        comm.Send(  wVec, dest=(iWork)+1, tag=2)
        comm.send(n_aVec, dest=(iWork)+1, tag=3)
        comm.Send(  aVec, dest=(iWork)+1, tag=4)
        if sameSeedForEachIndividual is False:
          comm.send(seed.item(i), dest=(iWork)+1, tag=5)
        else:
          comm.send(  seed, dest=(iWork)+1, tag=5)        
      else: # message size of 0 is signal to shutdown workers
        n_wVec = 0
        comm.send(n_wVec,  dest=(iWork)+1)
      i = i+1 
  
    # Get fitness values back for that batch
    i -= nSlave
    for iWork in range(1,nSlave+1):
      if i < nJobs:
        workResult = np.empty(hyp['alg_nVals'], dtype='d')
        comm.Recv(workResult, source=iWork)
        reward[i,:] = workResult
        i_steps = np.empty(1,dtype=int)
        i_steps = comm.recv(source=iWork,tag=6)
        steps -= i_steps
      i+=1
  return reward,steps

def slave():
  """Evaluation process: evaluates networks sent from master process. 

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node 
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """
  global hyp
  global the_game
  global full_actions
  global checkpoint
  #global the_ram
  outputsize = games_dict.get(the_game)
  games[hyp['task']] = games[hyp['task']]._replace(full = full_actions)
  if full_actions == False:  
    games[hyp['task']] = games[hyp['task']]._replace(output_size = outputsize)
    games[hyp['task']] = games[hyp['task']]._replace(o_act = np.full(outputsize,1))
  games[hyp['task']] = games[hyp['task']]._replace(env_name = the_game)
  #games[hyp['task']] = games[hyp['task']]._replace(ram = the_ram)
  task = WannGymTask(games[hyp['task']], nReps=hyp['alg_nReps'])
  
  # Evaluate any weight vectors sent this way
  while True:
    #print("NEW WHILE LOOP ITERATION")

    #print("New Iteration Worker",rank," playing",task.env.gname)
    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      start_time = time.time()
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights
      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it
      seed = comm.recv(source=0, tag=5) # random seed as int
      #print("wVEC",wVec.shape) #Atari (1225,) Racing (400,)
      #print("aVEC",aVec.shape) #Atari (35,) Racing (20,)
      #print("size of wVec",sys.getsizeof(wVec)/1e06,'MB')
      #print("size of wVec",sys.getsizeof(wVec)/1e06,'MB for worker', rank)
      #print("size of aVec",sys.getsizeof(aVec)/1e06,'MB for worker', rank)
      result,n_steps = task.getFitness(wVec,aVec,hyp,seed=seed) # process it
      n_steps = np.sum(n_steps)
      if hyp['task'].startswith("atari"):
        print("Worker",rank,"After getFitness result:",result,"in",n_steps,"steps, playing",task.env.gname,"--- %s seconds ---" % (time.time() - start_time))
      else:
        print("Worker",rank,"After getFitness result:",result,"--- %s seconds ---" % (time.time() - start_time))
      #print("--- %s seconds ---" % (time.time() - start_time))
      sending = int(n_steps)
      comm.Send(result, dest=0)            # send it back
      comm.send(sending, dest=0,tag=6)

    if n_wVec < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

def stopAllWorkers():
  """Sends signal to all workers to shutdown.
  """
  global nWorker
  nSlave = nWorker-1
  print('stopping workers')
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    #print('assigning the rank and nworkers', nWorker, rank)
    return "child"


# -- Input Parsing ------------------------------------------------------- -- #

def main(argv):
  """Handles command line input, launches optimization or evaluation script
  depending on MPI rank.
  """
  global fileName, hyp # Used by both master and slave processes
  global the_game, full_actions
  global checkpoint
  starting_time_total = time.time()
  hours = args.time
  checkpoint = args.infile
  '''global the_ram
  the_ram = args.ram'''
  full_actions = args.full
  the_game = args.game
  fileName    = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam
  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)
  # Launch main thread and workers
  if (rank == 0):
    master(starting_time_total,hours)
  else:
    slave()

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Evolve WANNs'))
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_wann.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default='p/laptop_swing.json')

  parser.add_argument('-o', '--outPrefix', type=str,\
   help='file name for result output', default='test_swing')
  
  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=2)
  
  parser.add_argument('-g', '--game', type=str,\
   help='which game to play', default='Phoenix')
  
  parser.add_argument('-f', '--full', type=bool,\
   help='full action space?', default=False)
  
  parser.add_argument('-t', '--time', type=int,\
   help='how many hours?', default=96)
  
  parser.add_argument('-i', '--infile', type=str,\
   help='file name for genome input', default='')

  args = parser.parse_args()


  # Use MPI if parallel
  if "parent" == mpi_fork(args.num_worker+1): os._exit(0)

  main(args)                              
  




