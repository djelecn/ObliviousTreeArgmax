import numpy as np



class ObliviousDecisionTree():
    
    def __init__(self, depth = 4):
        self.depth = depth
        
    
    def impurity_score(self,y1,y2):  
        if any((len(y1)<1,len(y2)<1)):
            score = 100
        else:
            score = ((y1-y1.mean())**2).mean() + ((y2-y2.mean())**2).mean()
        return score
    

    def get_split(self, x, y, groups_ids):         
        scores = []
        tresholds = []
        for i, v in enumerate(x):
            score = 0
            for g in range(len(groups_ids)):

                xg = x[groups_ids[g]]
                yg = y[groups_ids[g]]
                y1 = yg[xg<v]
                y2 = yg[xg>=v]

                impurity_score = self.impurity_score(y1,y2)
                score += impurity_score/len(groups_ids)
                
            scores.append(impurity_score)
            tresholds.append(v)
        return tresholds[np.argmin(scores)], min(scores)

    
    def get_best_split(self, X, y, groups_ids):  
        split_trs = []
        split_scores = []  
        for i in range(X.shape[1]):
            split_tr, split_score = self.get_split(X[:,i],y,groups_ids)
            split_trs.append(split_tr)
            split_scores.append(split_score)
        
        best_ind = np.argmin(split_scores)

        return best_ind, split_trs[best_ind], split_scores
    

    def perform_split(self, x, treshold, groups_ids):
        new_group_ids = []
        for gi in groups_ids:
            new_group_ids.append(gi[x[gi]<treshold])
            new_group_ids.append(gi[x[gi]>=treshold])
        
        return new_group_ids
    

    def fit(self, X, y, action_ids):    
        self.splits = []
        groups_ids = [np.arange(0,X.shape[0],1)]

        for d in range(self.depth):
            best_split_id, best_split_tr, leaf_errors = self.get_best_split(X, y, groups_ids)
            self.splits.append((best_split_id, best_split_tr))
            groups_ids = self.perform_split(X[:,best_split_id],best_split_tr,groups_ids)
        # for d in range(self.depth//2):
        #     best_split_id, best_split_tr, leaf_errors = self.get_best_split(X[:,action_ids], y, groups_ids)
        #     self.splits.append((best_split_id, best_split_tr))
        #     groups_ids = self.perform_split(X[:,best_split_id],best_split_tr,groups_ids)  

        self.leaf_counts = np.array([len(g) for g in groups_ids])
        self.leaf_errors = leaf_errors
        self.leaf_values = np.array([y[g].mean() if len(g)> 0 else 0 for g in groups_ids])
        

    def predict(self, X):
        groups_ids = [np.arange(0,X.shape[0],1)]
        preds = np.zeros(X.shape[0])
        for i, s in enumerate(self.splits):
            split_id, split_tr = s
            groups_ids = self.perform_split(X[:,split_id],split_tr, groups_ids)

        for i, g in  enumerate(groups_ids):
            if len(g)>0:
                preds[g] += self.leaf_values[i]       

        return preds
        



class ODT_agent():  
    def __init__(self, 
                 state_ids, 
                 action_format,
                 depth):      
        self.depth = depth
        self.tree = ObliviousDecisionTree(depth = depth)
        self.action_format = action_format
        self.action_dim = action_format[0]
        self.state_dim = len(state_ids)
        self.action_ids = np.arange(self.state_dim,self.state_dim+self.action_dim,1)
        self.state_ids = state_ids
        
        
    def tree_pass(self, obs):
        passes = [obs[:,i]>= tr if i in self.state_ids else 0 for (i, tr) in self.tree.splits]
        return passes
    

    def get_plausible_leaf_nodes(self, obs):
        d  = self.depth
        passes = self.tree_pass(obs)
        plausible_leaves = np.array([True]*(2**len(passes)))
        for i,pas in enumerate(passes[::-1]):
            if isinstance(pas,bool):
                new_possibilities = np.array(([not pas]*int(2**i)+[pas]*int(2**i))*int(2**(len(passes)-(i+1))))
                plausible_leaves *= new_possibilities     
        return plausible_leaves
    

    def get_state_value(self, obs):
        possible_leaves = self.get_plausible_leaf_nodes(obs)
        weights = self.tree.leaf_counts / self.tree.leaf_counts.sum()
        state_value = (possible_leaves * weights).sum()
        return state_value
    

    def get_best_leaf(self, obs):
        plausible_leaves = self.get_plausible_leaf_nodes(obs)
        return np.argmax(plausible_leaves*self.tree.leaf_values)
    

    def get_desired_path(self, obs):            
        best_leaf = self.get_best_leaf(obs)
        needed_passes = [best_leaf%(2**(self.depth-i))+1 > 2**(self.depth-i-1) for i in range(self.depth)]
        return needed_passes
    

    def get_best_action(self, obs):
        needed_passes = self.get_desired_path(obs)
        action_dic = {i:[self.action_format[1][0], self.action_format[1][1]] for i in self.action_ids} 
        for i, (f_i,tr) in enumerate(self.tree.splits):
            if f_i in self.action_ids:  
                if needed_passes[i]:
                    action_dic[f_i][0] = max(action_dic[f_i][0], tr)
                else:
                    action_dic[f_i][1] = min(action_dic[f_i][1], tr)
        return action_dic
    

    def act(self, obs, random = False):
#         actions = np.zeros(self.action_num)
        if random:   
            actions = np.random.randint(self.action_format[1][0], self.action_format[1][1],size=self.action_dim)[0]
        else:
            action_dic = self.get_best_action(obs)
            for i, (k, v) in enumerate(action_dic.items()):
                actions = np.random.randint(v[0],v[1], size = self.action_dim)[0]
        return actions
    

    def fit(self, X, y):
        self.tree.fit(X,y, self.action_ids)
