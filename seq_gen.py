# Generates sequences based on the provide distribtions

#TODO: Add noise

import numpy as np
import pickle

#vars
num_seq = 10000
out_file = 'gen_seq.txt'
mtrx_file = 'gen_mtrx.txt'
num_states = 4
num_obs = 3389

#randomly initialize matrices
#state transition matrix. Rows -> From & Columns -> To
trans = np.zeros((num_states, num_states))
#observation matrix. Rows -> State & Columns -> Observations
obs = np.zeros((num_states, num_obs))
for row in range(num_states-1):
    trans[row,1:] = np.random.dirichlet(np.ones(num_states-1),1)
for row in range(1,num_states):
    obs[row,:]=np.random.dirichlet(np.ones(num_obs),size=1)

start_state = 0
end_state = num_states-1

with open(out_file,'w') as f:
    for i in range(num_seq):
        seq=[]
        curr_state = start_state #set to start state
        while(curr_state!=end_state):
            #go to new state
            curr_state = np.random.choice(range(num_states), 1, replace=True, p=np.squeeze(trans[curr_state,:]))
            #get observation and add to sequence
            seq.append(np.random.choice(range(num_obs), 1, replace=True, p=np.squeeze(obs[curr_state,:]))[0])

        seq=map(str,seq)
        f.write("[,,,[\""+"\", \"".join(seq)+"\"]]\n") #write in form of other file

output = {"obs":obs, "trans":trans}
with open(mtrx_file,"wb") as f:
    pickle.dump(output, f)