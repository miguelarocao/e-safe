## EE/CS 159 Project
## Miguel Aroc-Ouellette

# TODO: Force final transition to end state
# Prevent premature transition to end state


import numpy as np
import random as rnd

class eSafe:
    def __init__(self, _num_states, _num_obs, prior_trans = None, prior_obs = None):
        """
        Constructor.
        Optional inputs for prior distributions.
        """

        #constant
        self.num_states =_num_states + 2                #+2 for start & end state
        self.num_obs = _num_obs
        self.learn_rate = 1                             #learning rate
        self.e = 0.8                                    #probability that algorithm picks safest (most probable) event
        self.start = 0                                  #start state index
        self.end = self.num_states - 1                  #end state index
        np.random.seed(123456)                          #set seed for reproducibility
        rnd.seed(123456)

        #var
        self.seq = None                                  #current observation sequence
        self.curr_state = None                           #current state
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

        if prior_trans:
            self.trans = prior_trans
        else:
            for row in range(self.num_states):
                self.trans[row,1:] = np.random.dirichlet(np.ones(self.num_states-1),1)

        if prior_obs:
            self.obs = prior_obs
        else:
            for row in range(self.num_states-1):
                self.obs[row,:]=np.random.dirichlet(np.ones(self.num_obs),size=1)

        return

    def train_on_seq(self):
        """ Trains on the current sequence. """

        for curr_ob in self.seq:
            new_state = None
            r_safe = rnd.uniform(0,1) < self.e
            if r_safe:
                #update the most probable event
                new_state = self.get_safe(curr_ob)
                print "Playing it safe on observation "+str(curr_ob)
            else:
                #pick an event at random from the possible events
                #   i.e. pick a random state, update trans and obs appropriately
                new_state = rnd.choice(range(1,self.num_states-1))
                print "Randomizing on observation "+str(curr_ob)
            print "Picked "+str(new_state)
            self.update_event(new_state, curr_ob)
            curr_state = new_state

            assert curr_state != self.start #should never be in start state after 1st transition

        #go to end state
        self.update_event(self.end,self.seq[-1])

        return

    def train(self, filename):
        """ Trains on the sequences found in the input file. Lazily read.
            Each sequence should occupy a new line and be commat separated."""

        with open(filename,'r') as f:
            while True:
                line = f.readline()
                if line=="":
                    break
                self.seq = map(int,line.strip().split(','))
                self.curr_state = self.start #reset to start state
                print "Training on: "+','.join(map(str,self.seq))
                self.train_on_seq()

        print "Transition Matrix: "
        print self.trans
        print "Observation Matrix: "
        print self.obs

    def get_safe(self,observation):
        """ Returns the most probable state given an observation, either simply
            witnessing the observation from the current state or transitioning and
            then witnessing the observation.
            Output: Index of most probable state."""

        safe = self.curr_state
        safe_prob = self.obs[safe,observation]*self.trans[safe, safe]
        print "State: "+str(safe)+" has prob "+str(safe_prob)
        for state in range(self.num_states):
            if state==self.curr_state:
                continue
            prob = self.obs[state,observation]*self.trans[self.curr_state, state]
            if prob > safe_prob:
                safe = state
                safe_prob = prob
            print "State: "+str(state)+" has prob "+str(prob)

        print "Safe state: "+str(safe)+" has prob "+str(safe_prob)
        return safe

    def update_event(self,state, observation):
        """ Updates the transition and observation matrices based on a new
            occurence of the input state & observation."""

        #update obs based, unless going to end state
        if state != self.end:
            self.update_distr(self.obs[state, :], observation, self.obs_count[state,:])
            self.obs_count[state, observation] += 1

        if state!=self.curr_state:
            #update state as well
            self.update_distr(self.trans[self.curr_state, :], state,
                                        self.state_count[self.curr_state,:])
            self.state_count[self.curr_state,state] +=1

        return

    def update_distr(self, distr, index, count_vec):
        """ Updates distribution given the update term, where the
                update is a value to be ADDED to the specified index.
            Update term uses the count_vec to modify learning rate by frequency;
                count_vec be indexed in the same way as the distribution."""

        length = len(distr)

        #update size. Function of learning rate and relative count
        #   The more frequently is seen, the lower the smaller the update
        update = self.learn_rate*(1 - count_vec[index]/sum(count_vec))

        if np.isnan(update): #handles division by zero
            update = self.learn_rate

        distr[index] *= 1 + update
        distr /= sum(distr)

        assert abs(sum(distr) - 1) < 1e-9 #allows for rounding errors

        return

def main():
    # Test on sample
    my_alg = eSafe(3,5)
    my_alg.train("sample_data.txt")

if __name__ == '__main__':
    main()
