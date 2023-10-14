import numpy as np



class Datalog():
    
    def __init__(self, gamma = 0.3):
        self.data = None
        self.gamma = gamma
        

    def update(self, observation, actions, reward,state_value, rollout_id):
        bellman_value = reward + self.gamma*state_value
        new_data = np.concatenate([observation, actions, reward, bellman_value, rollout_id]).reshape(1,-1)
        if self.data is None: 
            self.data = new_data
        else:
#             self.data[self.data[:,-1] == rollout_id][:,-2] = (1-self.gamma)*self.data[self.data[:,-1] == rollout_id][:,-3] + self.gamma*reward
            self.data = np.concatenate([self.data,new_data], axis = 0)   
        if self.data.shape[0]>100000:
            self.data = np.delete(self.data, [0], axis = 0)


    def reset(self): 
        del self.data
        self.data = None
          

    @property        
    def X(self):
        return self.data[:,:-3]
    

    @property
    def y(self):
        return self.data[:,-2]