import numpy as np

# class SimpleReward():
#     def __init__():
#         self.goal = 10
        
#     def reward(self):
        

class RewardCircle():
    def __init__(self):
        self.v_goal     =   1
        self.c_world    =   -0.75
        self.c_robot    =   -1
        self.d_pos      =   0.01
        self.d_neg      =   -0.002
        self.a_pos      =   0.001
        self.a_neg      =   -0.0002
        self.l_pos      =   0.05
        self.l_neg      =   -0.01
        self.w_neg      =   -0.01
        
        self.threshold  =   20
        self.w_dir      =   np.pi/12
        self.r_col      =   60
        
    def reward(self, agent, goal):
        # return 0
        history = agent.history.get()
        # collision (world, agent), d_0, r_collision, l_laser
        if history["od"][-1] < self.threshold:
            return self.v_goal
        if agent.collision_w:
            return self.c_world
        if agent.collision_a:
            return self.c_robot
        
        # r_dist
        r_dist = history["od"][-2] - history["od"][-1]
        r_dist = r_dist*self.d_neg if r_dist < 0 else r_dist*self.d_pos
        
        # r_ori   
        # print(history["og"][-2:], history["og"][-4:-2], history["og"].shape)
        cross = np.linalg.norm(np.cross(history["og"][-2:], history["og"][-4:-2]))
        dot_p = np.dot(history["og"][-1], history["og"][-2])
        a_norm = 1 - 2*np.arctan2(cross, dot_p)/np.pi
        r_ori = a_norm*self.a_neg if a_norm < 0 else a_norm*self.a_pos
        
        # r_st
        r_st = (history["d_0"] - history["od"][-1])*self.l_pos if history["d_0"] > history["od"][-1] else 0
        
        # r_mld
        if min(history["ol"][-history["ol"].shape[0]//agent.history.lenght:]) < self.r_col:
            r_mld = (self.r_col + 500 - min(history["ol"][-agent.lasers:]))*(self.l_neg)
        else:
            r_mld = 0
        
        penalty = 0
        if np.abs(history["ov"][-2]) < 0.001:
            penalty = -0.0001
        
        # r_wig
        r_wig = 0 
        return r_dist+r_ori+r_st+r_mld+r_wig+penalty


class MyReward:
    def __init__(self):
        self.goal_reward = 10
        self.reach_reward = 100
        self.penalty = -10
    
         
    def reward(self, agent, goal):
        his = agent.history.get()
        op= his["op"]
        ol= his["ol"]
        # print(np.linalg.norm(op[:2]))
        prev = np.linalg.norm(op[-2][:-1])
        cur = np.linalg.norm(op[-1][:-1])
        if min(ol) < 50:
            return self.penalty / (min(ol)/50)
        return (self.goal_reward)*((prev-cur)/agent.v_limit)
        # if min()
        # raise
        return 0