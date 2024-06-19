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
        self.goal_reward = 1
        self.reach_reward = 10
        self.penalty = 1
        self.history = {}
    
         
    def reward(self, agent, goal):
        his = agent.history
        op= his[16*4:]
        ol= his[:16*4]
        rew = 0
        r_dist, p_zone, p_col, rew = 0, 0, 0, 0

        if agent not in self.history:
            self.history[agent] = np.linalg.norm(goal-agent.position)
            r_dist = 0
        else:
            r_dist = self.goal_reward*(self.history[agent] - np.linalg.norm(goal-agent.position))/agent.v_limit
            self.history[agent] = np.linalg.norm(goal-agent.position)
        
        if min(ol[-16:]) * agent.laser_lenght < agent.save_zone:
            p_zone = self.penalty*((min(ol[-16:]) * agent.laser_lenght - agent.save_zone)/agent.save_zone)
        else:
            p_zone = 0
            
        if agent.collision_w or agent.collision_a:
            p_col = -1
        else:
            p_col = 0
            
        if agent.reached:
            rew = 2
        else:
            rew = 0
            
        return r_dist + p_zone + p_col + rew
 
class MyReward2:
     def __init__(self):
         self.goal_dir_rew   = 1
         self.goal_reach_rew = 2
         self.staying_pen    = -0.5
         self.dist_goal_rew  = 2
         self.lasers_penalty = -0.5
         self.collision_ag   = -1.2
         self.collision_wo   = -1
         self.history        = {}
         self.init_dists     = {}

     def reward(self, agent, goal):
         his = agent.history
         op = his[16*4:]
         ol = his[:16*4]

         if agent.reached:
             return self.goal_reach_rew

         if agent.collision_w:
             return self.collision_wo
         elif agent.collision_a:
             return self.collision_ag

         rew = 0
         if agent not in self.history:
             self.history[agent] = np.linalg.norm(goal-agent.position)
             self.init_dists[agent] = np.linalg.norm(goal-agent.position)
             return 0

         rew += self.goal_dir_rew*(self.history[agent] - np.linalg.norm(goal-agent.position))/agent.v_limit
         self.history[agent] = np.linalg.norm(goal-agent.position)
         
         if np.abs(rew) <= 0.01:
            return self.staying_pen
        
         rew += self.dist_goal_rew*(self.init_dists[agent] - np.linalg.norm(goal-agent.position))/self.init_dists[agent]

         if min(ol[-16:]) * agent.laser_lenght < agent.save_zone:
            rew += self.lasers_penalty*((min(ol[-16:]) * agent.laser_lenght - agent.save_zone)/agent.save_zone)

         return rew
