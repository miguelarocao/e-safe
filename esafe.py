## EE/CS 159 Project
## Miguel Aroc-Ouellette

# TODO: Improve Learning rate formula
# TODO: Find appropriate leraning rate
# TODO: Find appropriate # of hidden states
# TODO: Find appropriate e (safe probaility variable)
# TODO: Improve rank comparison, maybe group by 10 minutes and take average?
# TODO: Add distribution based comparison

import numpy as np
import random as rnd
import datetime
import matplotlib.pyplot as plt
import pickle

from naive_online import Naive, TrainingInterface

class eSafe:
    ### Initialization Methods ###
    def __init__(self, _num_states, _num_obs,rt ,e1, prior_trans = None, prior_obs = None,  ):
        """
        Constructor.
        Optional inputs for prior distributions.
        """
        # rt will be miultiplied by 1/3389 to generate the learning rate
        # e1 is the safe probability 
        #constant
        self.num_states =_num_states + 2                #+2 for start & end state
        self.num_obs = _num_obs
        self.learn_rate = rt/_num_obs                  #learning rate
        print "learn rate" + str(rt)
        print "e" + str(e1)
        self.e = e1                                    #probability that algorithm picks safest (most probable) event
        self.start = 0                                  #start state index
        self.end = self.num_states - 1                  #end state index
        np.random.seed(123456)                          #set seed for reproducibility
        rnd.seed(123456)

        #var
        self.curr_state = None                                #current state
        self.obs_count = np.zeros([self.num_states,_num_obs]) #count for observations
        self.state_count = np.zeros([self.num_states]*2)      #count for transitions

        #-- Set up matrices
        #state transition matrix. Rows -> From & Columns -> To
        self.trans = np.zeros((self.num_states, self.num_states))
        #observation matrix. Rows -> State & Columns -> Observations
        self.obs = np.zeros((self.num_states, self.num_obs))
        self.init_matrices(prior_trans, prior_obs)

    def init_matrices(self, prior_trans = None, prior_obs = None):
        """ Initialiazes the matrices randomly, or with prior information"""

        if prior_trans is not None:
            self.trans = prior_trans
        else:
            for row in range(self.num_states-1):
                self.trans[row,1:] = np.random.dirichlet(np.ones(self.num_states-1),1)

        if prior_obs is not None:
            self.obs = prior_obs
        else:
            for row in range(1,self.num_states):
                self.obs[row,:]=np.random.dirichlet(np.ones(self.num_obs),size=1)

        return

    ### TRAINING METHODS ###

    def train_on_seq(self, seq):
        """ Trains on the current sequence. """

        self.curr_state = self.start #reset to start state

        for curr_ob in seq[:-1]: #for everything but last state
            new_state = None
            r_safe = rnd.uniform(0,1) < self.e
            if r_safe:
                #update the most probable event
                new_state = self.get_safe(curr_ob)
                #print "Playing it safe on observation "+str(curr_ob)
            else:
                #pick an event at random from the possible events
                #   i.e. pick a random state, update trans and obs appropriately
                new_state = rnd.choice(range(1,self.num_states-1))
                #print "Randomizing on observation "+str(curr_ob)
            #print "Picked "+str(new_state)
            self.update_event(new_state, curr_ob)
            self.curr_state = new_state

            assert self.curr_state != self.start #should never be in start state after 1st transition

        #go to end state
        self.update_event(self.end,seq[-1])

        return

    def train(self, filename, num_seq):
        """ NOTE: Deprecated. Currently use TrainInterface.
        Trains on the sequences found in the input file. Lazily read."""

        count=0
        with open(filename,'r') as f:
            for line in f:
                seq = line.rstrip()[1:-1].split(',')[data_dict['seq']:]
                seq[0] = seq[0][1:]
                seq[-1] = seq[-1][:-1]
                seq = [int(x[2:-1]) for x in seq]
                #print "Training on: "+','.join(map(str,self.seq))\

                #evaluate sequence BEFORE training
                eval_success = self.eval_wrapper(seq)

                #train
                self.train_on_seq(seq)

                #only increment counter if evaluation was succesful
                if eval_success:
                    count+=1
                    if count%100==0:
                        print "Processed "+str(count)+"..."
                    if count >= num_seq:
                        break

        self.eval_plot()
        #print "Transition Matrix: "
        #print self.trans
        #print "Observation Matrix: "
        #print self.obs

        return

    def get_safe(self,observation):
        """ Returns the most probable state given an observation, either simply
            witnessing the observation from the current state or transitioning and
            then witnessing the observation.
            Output: Index of most probable state."""

        safe = self.curr_state
        safe_prob = self.obs[safe,observation]*self.trans[safe, safe]
        #print "State: "+str(safe)+" has prob "+str(safe_prob)
        for state in range(self.num_states-1): #can't go to end state!
            if state==self.curr_state:
                continue
            prob = self.obs[state,observation]*self.trans[self.curr_state, state]
            if prob > safe_prob:
                safe = state
                safe_prob = prob
            #print "State: "+str(state)+" has prob "+str(prob)

        #print "Safe state: "+str(safe)+" has prob "+str(safe_prob)
        return safe

    def update_event(self, state, observation):
        """ Updates the transition and observation matrices based on a new
            occurence of the input state & observation."""

        #update observation distribution
        self.update_distr(self.obs[state, :], observation, self.obs_count[state,:])
        self.obs_count[state, observation] += 1

        #update state
        self.update_distr(self.trans[self.curr_state, :], state, self.state_count[self.curr_state,:])
        self.state_count[self.curr_state, state] +=1

        return

    def update_distr(self, distr, index, count_vec):
        """ Updates distribution given the update term, where the
                update is a value to be ADDED to the specified index."""

        #length = len(distr)

        #update size. Function of learning rate and relative count
        #   The more frequently is seen, the lower the smaller the update
        #update = self.learn_rate*(1 - count_vec[index]/sum(count_vec))
        update = self.learn_rate

        if np.isnan(update): #handles division by zero
            update = self.learn_rate

        distr[index] += update
        distr /= sum(distr)

        assert abs(sum(distr) - 1) < 1e-9 #allows for rounding errors

        return

    ### EVALUATION METHODS ###

    def eval_seq(self, seq):
        """Walks through most probable sequence state and evaluates at each step
        how well the algorithm does at predicting the next obvservation.
        Input: seq is the observation sequence.
        Output: Rank offset (as a percent of total number of observations) per sequence step.
                i.e. if next observation is ranked as 10, and there are 200 possible observations
                then at that time step the relative rank is 10/200=0.05.
                Therefore, low value is GOOD, high is BAD. BEST is rank 0 (i.e. most probable)."""

        # Output Notes: Index 0 is predicting the first observation given no data
        #               Index 1 is predicting the second observation given the first
        #               Index n is predicting the (n+1)th observation given n observations
        # Note: (n)th observation = seq[n-1]
        offset = [0]*(len(seq))

        for n in range(len(seq)):
            #get most probable current state given the sequence we've seen thus far
            if n != self.start:
                _, curr_state = self.get_prob_seq(seq[:n],False) #up to, but excluding i
            else:
                curr_state = self.start

            #get the ranked most probable observations
            prob_rank = self.get_obs_rank(curr_state)

            #check rank
            offset[n] = np.where(prob_rank==seq[n])[0][0]/(self.num_obs*1.0)

        return offset

    def get_obs_rank(self, curr_state):
        """Returns the next pages ranked by probability of occurence given the current state."""

        prob = [0]*self.num_obs
        for state in range(self.num_states):
            #iterate through each state and take max probability on each page
            prob_next = np.multiply(self.obs[state,:], self.trans[curr_state,state])
            prob = np.array([(prob[i]+prob_next[i]) for i in range(self.num_obs)])

        #sort in decreasing probability
        top = prob.argsort()[::-1]

        return top

    def get_prob_seq(self, seq, end=True):
        """Returns the probability of a given observation sequence.
        Uses Viterbi Algorithm and Dynamic Programming.
        Optinal input denotes if we should finish on the end state or not,
        if not then function also outputs last state."""

        #small log numbers
        log_zero = -1e20

        #append None for start state
        seq = list([None]+seq)

        # input sequence length
        seq_len = len(seq)

        # stores P(Best sequence)
        prob = [[[0] for i in range(self.num_states)] for j in range(seq_len)]

        # Stores most likely hidden state sequence
        state_seq = [[[''] for i in range(self.num_states)] for j in range(seq_len)]

        # always start in start state (in log form
        prob[0] = [log_zero]*self.num_states
        prob[0][self.start] = 0

        # initalize best sequence of length 1
        state_seq[0] =[str(i) for i in range(self.num_states)]

        # iterate through all observations in given sequence
        for length in range(1, seq_len):    #skip initial state
            for state in range(self.num_states):
                max_state = 0
                best_prob = log_zero #can't be 0 due to log

                # iterate through all possible transitions
                for prev in range(self.num_states):
                    # cur_prob is the probability of transitioning to 'state'
                    # from 'prev' state and observing the correct state.
                    if self.trans[prev][state]==0 or self.obs[state][seq[length]]==0:
                        cur_prob = log_zero #handles log(0)
                    else:
                        cur_prob = prob[length - 1][prev] + np.log(self.trans[prev][state]) + np.log(self.obs[state][seq[length]])

                    if cur_prob > best_prob:
                        max_state, best_prob = prev, cur_prob

                    #print "("+str(length)+") "+str(prev)+"->"+str(state)+": "+str(np.exp(cur_prob))

                    # update best probability
                    prob[length][state] = best_prob
                    # update sequence
                    state_seq[length][state] = state_seq[length - 1][max_state] + str(state)

            prob[length] = prob[length][:]   # copies by value
            state_seq[length] = state_seq[length][:]

        if not end:
            max_ind = 0
            for i in range(self.num_states):  # find most-likely index of entire sequence
                if prob[seq_len - 1][i] > prob[seq_len - 1][max_ind]:
                    max_ind = i

            #output (probability, last state)
            return (np.exp(prob[-1][max_ind]), int(state_seq[-1][max_ind][-1]))

        # get maximum sequence that brought you to end state
        #print state_seq[-1][self.end][1:]

        # return result. Note, always end on end state!
        return np.exp(prob[-1][self.end])

    ### I/O METHODS ###

    def dump_distr(self, filename):
        """ Pickles the observation and transition matrices to the specfied file."""
        output = {"obs":self.obs, "trans":self.trans}
        with open(filename,"wb") as f:
            pickle.dump(output, f)

    def load_distr(self, filename):
        """ Unpickles the observation and transition matrices from the specified file."""
        with open(filename, "rb") as f:
            in_dict = pickle.load(f)
            self.trans = in_dict["trans"]
            self.obs = in_dict["obs"]

def plot_by_seq_len(self, learner, length_range):
    pass

def main():
    # Test on sample
 #   path="D:\\Datasets\\ML_Datasets\\seq_data\\"

    #path="D:/Assignments/CS 159/e-safe"
    path=""
    fname='data_2_1.txt' #max value is 3388
#
    #KL Distribution divergence example
#    safelearner = eSafe(4,3389)
#    with open('hmm_base_10k.dat','r') as f:
#        data_hmm = pickle.load(f)   #previously trained on first 10k sequences in data_2_1.txt
#    mydict = {"safe":safelearner}   
#    mydict["safe2"] = safelearner2
#    mytrainer = TrainingInterface(mydict)
#    mytrainer.set_eval_mode("Kullback-Leibler",data_hmm)
#    mytrainer.train(path+fname,1000)

##    #Rank Offset Example
#    for rate in [.3, .5, .6 ]:
#        for e in [.75, .8, .85]    :

    mylearner = Naive(3389)
    mydict = {"naive":mylearner}
    #for state in [5,7,9]:
    for rate in [.3]:
        state = 9
        #rate = .3
        e = .8
        #state = 13
        safelearner = eSafe(state,3389, rate, e)
        mydict["safe" + str(rate)] = safelearner
        print "with state" + str(state)
    mytrainer = TrainingInterface(mydict)
    mytrainer.set_eval_mode("Rank Offset",4)
    mytrainer.train(path+fname,10000, 10, 2,100)
    safelearner.dump_distr("safe_mtrx.dat")


##    myfile= open(path+fname)
##    count = 0
##    run_max = 0
##    run_min = 10000
##    for line in myfile:
##        seq = line.rstrip()[1:-1].split(',')[data_dict['seq']:]
##        seq[0] = seq[0][1:]
##        seq[-1] = seq[-1][:-1]
##        seq = [int(x[2:-1]) for x in seq]
##        curr_max = max(seq)
##        curr_min = min(seq)
##        if curr_max > run_max:
##            run_max = curr_max
##        if curr_min < run_min:
##            run_min = curr_min
##        count +=1
##        if count%10000==0:
##            print count
##    print count
##    print "Max: "+str(run_max)
##    print "Min: "+str(run_min)

##    #two states healthy (0) and fever(1) -> health (1) and fever (2)
##    #three observations dizzy (0), cold (1) and normal (2)
##    trans = np.array([[0,0.6,0.4,0],
##                      [0,0.6,0.3,0.1],
##                      [0,0.3,0.6,0.1],
##                      [0,0,0,0]])
##    obs = np.array([[0,0,0],
##                    [0.1,0.4,0.5],
##                    [0.6,0.3,0.1],
##                    [0.5,0.4,0.1]])
##
##    mylearner=eSafe(2,3,trans,obs)
##
##    #viterbi test
##    print mylearner.get_prob_seq([2],False)

if __name__ == '__main__':
    main()
