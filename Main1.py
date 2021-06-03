import numpy as np
import csv
from numpy  import array
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from SVM_Classifier import SVM_Classifier

inputdata = []
with open("wpbc.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        inputdata.append(row)
Data = array(inputdata)
num_features = Data.shape[1] - 1
   
# initialization
num_episodes = 1000
#num_states = 2 ** num_features
num_actions = num_features
AOR = np.zeros([num_features,3]) # store sumation rewards for an action
num_visitedstates = [] # store number of visited states
steps = np.zeros(num_episodes,dtype=int) # store #steps in each episode
rewards = np.zeros(num_episodes) # store total rewards for each episode

epsilon = 0.3
alpha = 0.3
gamma = 1

V = np.zeros([1,2]) # store state-values
V_row = np.zeros([1,2])
Q = np.zeros([1,num_features + 1])
Q_row = np.zeros([1,num_features + 1])
num_visit = np.zeros([1,2]) # store #visits of each states
num_visit_row = np.zeros([1,2])
Q[0,0] = -1
V[0,0] = -1
num_visit[0,0] = -1
index = np.zeros([1,1])
index_row = np.zeros([1,1])

    
for ep in range(num_episodes):
    print(ep)
    steps[ep] += 1
    s = 0#round(np.random.rand()* (2 ** num_features - 1))
    #while(s == 2 ** num_features - 1):
    #    s = round(np.random.rand()* (2 ** num_features - 1))
    z = np.argwhere(index[:,0] == s)
    #print(z.shape[0])
    if (z.shape[0] == 0):
        index_row[0,0] = s
        index = np.vstack([index, index_row])
        num_visit_row[0,0] = s
        num_visit_row[0,1] = 1
        num_visit = np.vstack([num_visit, num_visit_row])
        #Q_row = np.zeros([1,num_features + 1])
        #Q_row[0,0] = s
        #Q = np.vstack([Q, Q_row])
        V_row[0,0] = s
        V = np.vstack([V, V_row])
    else:
        num_visit[z[0][0],] += 1
    z = np.argwhere(index == s)
    current_s_bin = list(bin(s)[2:].zfill(num_features))
    current_cr = SVM_Classifier(current_s_bin, Data)
    available_actions = []
    Q_cur_actions = []
    for i in range(num_features):
        if (current_s_bin[i] == '0'):
            available_actions.append(i)
            #Q_cur_actions.append(Q[z[0][0],i+1])
            Q_cur_actions.append(AOR[i,2])
    if (z.shape[0] != 0):
        if(np.random.rand()<epsilon):
            rnd = np.random.randint(len(available_actions))
            a = available_actions[rnd]
        else:
            index_max = Q_cur_actions.index(max(Q_cur_actions))
            a = available_actions[index_max]  
    else:
        rnd = np.random.randint(len(available_actions))
        a = available_actions[rnd]
    while (True):
        next_s_bin = list(bin(s)[2:].zfill(num_features))
        next_s_bin[a] = '1'
        next_cr = SVM_Classifier(next_s_bin, Data)
        R = next_cr - current_cr
        rewards[ep] += R
        AOR[a,0] += R
        AOR[a,1] += 1
        AOR[a,2] = AOR[a,0]/AOR[a,1]
        next_s_str = ''.join(next_s_bin)
        s_prime = int(next_s_str,2)
        z_prime = np.argwhere(index == s_prime)
        if (z_prime.shape[0] == 0):
            index_row = s_prime
            index = np.vstack([index, index_row])
            num_visit_row[0,0] = s_prime
            num_visit_row[0,1] = 1
            num_visit = np.vstack([num_visit, num_visit_row])
            #Q_row = np.zeros([1,num_features + 1])
            #Q_row[0,0] = s_prime
            #Q = np.vstack([Q, Q_row])
            V_row[0,0] = s_prime
            V = np.vstack([V, V_row])
        else:
            num_visit[z_prime[0][0],1] += 1
        z_prime = np.argwhere(index == s_prime)
        next_actions = []
        Q_nex_actions = []
        for i in range(num_features):
            if (next_s_bin[i] == '0'):
                next_actions.append(i)
                #Q_nex_actions.append(Q[z_prime[0][0],i+1])
                Q_nex_actions.append(AOR[i,2])
        #print([s,a,ep,next_actions])
        if (s_prime == 2 ** num_features - 1):
            z = np.argwhere(num_visit[:,0] == s)
            z = z[0][0]
            #Q[z,a] += alpha * (R - Q[z,a])
            V[z,1] += alpha * (R - V[z,1])
            break
        else:
            if (z_prime.shape[0] == 0):
                rnd = np.random.randint(len(next_actions))
                a_prime = next_actions[rnd]
            else:
                if (np.random.rand()<epsilon):
                    rnd = np.random.randint(len(next_actions))
                    a_prime = next_actions[rnd]
                else:
                    index_max1 = Q_nex_actions.index(max(Q_nex_actions))
                    a_prime = next_actions[index_max1]
            z = np.argwhere(index == s)
            z_prime = np.argwhere(index == s_prime)
            z = z[0][0]
            z_prime = z_prime[0][0]
            #Q[z,a+1] += alpha * (R + gamma * Q[z_prime,a_prime + 1] - Q[z,a + 1])
            V[z,1] += alpha * (R + gamma * V[z_prime,1] - V[z,1])
            s = s_prime
            a = a_prime
            steps[ep] += 1
            rewards[ep] += R
    num_visitedstates.append(sum(x > 0 for x in num_visit))

num_visitedstates = np.asarray(num_visitedstates)

np.savetxt("AOR.csv", AOR, delimiter=",")
np.savetxt("Q.csv", Q, delimiter=",")
np.savetxt("V.csv", V, delimiter=",")
np.savetxt("rewards.csv", rewards, delimiter=",")
np.savetxt("steps.csv", steps, delimiter=",")
np.savetxt("num_visitedstates.csv", num_visitedstates, delimiter=",")
np.savetxt("num_visit.csv", num_visit, delimiter=",")
