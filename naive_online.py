#def naive learner class

import numpy as np
import random as rnd
import matplotlib.pyplot as plt


data_dict={'user_id':0,
           'start_time':1,
           'end_time':2,
           'seq':3}

class TrainingInterface:
    """Allows you to train on multiple models simultaneously"""

    def __init__(self, models):
        """ Constructors.
        Inputs:
            {models} should be a dict of models to train & evaluate.
            Models MUST have a train() and the appropriate eval methods."""

        if type(models)!=dict:
            print "Invalid input! Please provide a dict of models to train on."

        self.models = models
        self.eval_results = [[] for i in range(len(models))]

        self.eval_mode = "None"                          #evaluation mode. set with set_eval_mode()
        self.eval_param = None                           #evaluation parameters
        self.eval_output = {name:[] for name in models.iterkeys() } #evaluation output, index by modelname

    def train(self, filename, eval_length, eval_start = 0, eval_step = 1):
        """Trains on the sequences in the files. Lazy reading.
        Inputs:
            filename is the file name.
            eval_length is the number of sequences to evaluate on. Trains on at least this many sequences.
            (Optional)
            eval_start is the sequence number to start evaluating on.
            eval_step dictates the number of training sequences between each evaluation seq.
        """
        count = 0
        eval_count = 0
        with open(filename,'r') as f:
            for line in f:
                seq = self.parse_line(line)
                success = False
                #evaluate all models
                if count >= eval_start and (count%eval_step) == 0:
                    success = self.eval_wrapper(seq)

                #train each model
                for model in self.models.itervalues():
                    model.train_on_seq(seq)

                if success:
                    eval_count +=1
                    if eval_count >= eval_length:
                        break

                if count%100 == 0:
                        print "Trained on "+str(count)+" Evaluated on "+str(eval_count)+"..."
                count +=1

        #plot
        xvals = np.arange(eval_start, eval_start + eval_length*eval_step, eval_step)
        self.eval_plot(xvals)

    def parse_line(self, line):
        """ Parses line """
        seq = line.rstrip()[1:-1].split(',')[data_dict['seq']:]
        seq[0] = seq[0][1:]
        seq[-1] = seq[-1][:-1]

        return [int(x[2:-1]) for x in seq]

    def set_eval_mode(self, mode, input_param = None):
        """Set the evaluation mode and input parameters."""
        self.eval_mode = mode
        self.eval_param = input_param


        if self.eval_mode == "Rank Offset":
            # Input parameter is the maximum sequence length to consider (inclusive)
            # Output is plot of offset for each sequence position

            if input_param is None:
                print "Please supply max sequence length!"
                raise (AssertionError)

            for name in self.models.iterkeys():
                self.eval_output[name] = [[] for _ in range(input_param)]

    def eval_wrapper(self, seq):
        """Evaluates current sequence for models based on different modes.
           Returns boolean of success."""

        if self.eval_mode == "None":
            return True
        elif self.eval_mode == "Rank Offset":
            seq_len = len(seq)

            if seq_len < self.eval_param:
                return False

            for name,model in self.models.iteritems():
                prob_seq = model.eval_seq(seq[:self.eval_param])
                for i in range(self.eval_param):
                    self.eval_output[name][i] += [prob_seq[i]]

        return True

    def eval_plot(self, xvals):
        """Prints data for different evaluation modes."""
        plt.style.use('ggplot')

        if self.eval_mode == "None":
            return
        elif self.eval_mode == "Rank Offset":
            for i in range(self.eval_param):
                for name, result in self.eval_output.iteritems():
                    plt.plot(xvals, result[i], label = name + " Pos#" + str(i))
                plt.plot([0,xvals[-1]], [0.5, 0.5],
                            label = "Random Baseline", color='k', linestyle='-', linewidth=2)
                plt.ylabel('Rank Offset Percentile')
                plt.xlabel('Sequence Count')
                plt.title('Rank Offset for Sequences of Length '+str(self.eval_param))
                plt.legend()
                plt.show()

class Naive:
    ### Initialization methods ###
    def __init__(self, _num_obs):
        """ Constructor """

        #constant
        self.num_obs = _num_obs
        #the rate at which counts decay, in [0,1] where closer to 0 forgets faster
        self.forget_rate = 0.95
        rnd.seed(123456)

        #var
        self.obs_count = np.zeros(self.num_obs)

    def train_on_seq(self, seq):
        """ Train on the current sequence.
            Aka. update count and forget """

        for obs in seq:
            # forget
            self.obs_count = np.multiply(self.forget_rate*1.0, self.obs_count)

            # update based on new observation
            self.obs_count[obs]+=1

        return

    def eval_seq(self, seq):
        """Ranks the most probably observations based on the naive learner."""

        # Output Notes: Index 0 is predicting the first observation given no data
        #               Index 1 is predicting the second observation given the first
        #               Index n is predicting the (n+1)th observation given n observations
        # Note: (n)th observation = seq[n-1]
        offset = [0]*(len(seq))

        for n in range(len(seq)):
            #get the ranked most probable observations
            naive_rank = self.get_naive_rank()

            #check rank
            offset[n] = np.where(naive_rank==seq[n])[0][0]/(self.num_obs*1.0)

        return offset

    def get_naive_rank(self):
        """ Uses the observation count to rank the most probable outputs. """

        counts = list(self.obs_count)

        #perturb by small deviations to break ties (i.e. all 0s)
        counts += np.random.rand(self.num_obs)/1e10

        return counts.argsort()[::-1]

    ### I/O METHODS ###

    def dump_distr(self, filename):
        """ Pickles the observation matrix from the specfied file."""
        with open(filename,"wb") as f:
            pickle.dump(self.obs, f)

    def load_distr(self, filename):
        """ Unpickles the observation matrix from the specified file."""
        with open(filename, "rb") as f:
            self.obs = pickle.load(f)

def main():
        # Test on sample
    path="D:\\Datasets\\ML_Datasets\\seq_data\\"
    fname='data_2_1.txt' #max value is 3388
    #mylearner=eSafe(4,3388)
    #mylearner.set_eval_mode("Rank Offset",2)
    #mylearner.train(path+fname,10)

    mylearner = Naive(3389)
    mylearner2 = Naive(3389)
    mylearner2.forget_rate = 0.5
    mytrainer = TrainingInterface({"naive":mylearner, "forgetful":mylearner2})
    mytrainer.set_eval_mode("Rank Offset",2)
    mytrainer.train(path+fname,100, 10, 2)
    pass

if __name__ == '__main__':
    main()
