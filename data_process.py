# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:39:16 2024

@author: 15834
"""

import numpy as np
import pandas as pd
import math
import warnings
from data_utils import agent_norm, map_norm
from scenario_visualization import scenario_viz
import gc
warnings.filterwarnings('ignore')

# Data process
class DataProcess(object):
    def __init__(self, data1):
        self.num_neighbors = 3
        self.hist_len = 40
        self.future_len = 10
        self.map_length = 50
        self.max_neighbor_distance = 50
        self.data = data1
        #self.data = pd.concat([data1,data2,data3],ignore_index=True)
        self.data = self.data[self.data["Lane_ID"]<=6].reset_index(inplace=False)
        self.data = self.unitConversion(self.data)
        self.map_dict = self.build_map() 
    
    def unitConversion(self,df):
        '''
        transform to m
        :param df: df
        :return: transformed df
        '''
        ft_to_m = 0.3048
        df['Global_Time'] = df['Global_Time']
        for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_Length", "v_Width"]:
            df[strs] = df[strs] * ft_to_m
        df["v_Vel"] = df["v_Vel"] * ft_to_m*3.6
        df["v_Acc"] = df["v_Acc"] * ft_to_m
        return df
    
    def cal_heading(self,group):
        '''
        grouped data by ID
        calculate heading, velociity, acceleration
        '''
        group["Heading"] = 0
        group['velocity_x'] = 0
        group['velocity_y'] = 0
        group['acc_x'] = 0
        group['acc_y'] = 0
        for i in range(0,len(group)-1):
            delta_y = group.loc[i+1].Local_X - group.loc[i].Local_X
            delta_x = group.loc[i+1].Local_Y - group.loc[i].Local_Y
            if delta_x == 0:
                if i == 0:
                    group.loc[i,"Heading"] = 0
                else:
                    group.loc[i,"Heading"] = group.loc[i-1,"Heading"]
            else:
                group.loc[i,"Heading"] = np.arctan(delta_y/delta_x)
        for i in range(0,len(group)):
            group.loc[i,'velocity_x'] = group.loc[i,"v_Vel"]*math.sin(group.loc[i,"Heading"])
            group.loc[i,'velocity_y'] = group.loc[i,"v_Vel"]*math.cos(group.loc[i,"Heading"])
            group.loc[i,'acc_x'] = group.loc[i,"v_Acc"]*math.sin(group.loc[i,"Heading"])
            group.loc[i,'acc_y'] = group.loc[i,"v_Acc"]*math.cos(group.loc[i,"Heading"])
        return group
    
    def ego_process(self, group, timestep):
        ego_states = np.zeros(shape=(self.hist_len, 7))
        # get the sdc current state
        self.current_xyh = np.array((group.loc[timestep].Local_Y, group.loc[timestep].Local_X, group.loc[timestep].Heading))
        # add sdc states into the array
        for i in range(timestep+1-self.hist_len,timestep+1):
            ego_state = np.array([group.loc[i].Local_Y, group.loc[i].Local_X, group.loc[i].Heading, group.loc[i].velocity_y,
                                  group.loc[i].velocity_x, group.loc[i].acc_y, group.loc[i].acc_x])
            ego_states[i-timestep-1+self.hist_len] = ego_state
        # size = (20,7)
        return ego_states.astype(np.float32)
    
    def neighbors_process(self,sdc_id, group, timestep):
        neighbors_states = np.zeros(shape=(self.num_neighbors, self.hist_len, 7))
        neighbors = {}
        self.neighbors_id = {}
        start_time = group.loc[timestep+1-self.hist_len].Global_Time
        current_time = group.loc[timestep].Global_Time
        ego_lane = group.loc[timestep].Lane_ID
        # search for nearby agents
        search_data = self.data[self.data["Global_Time"]==current_time].reset_index(inplace=False)
        for i in range(0,len(search_data)):
            if search_data.loc[i].Vehicle_ID != sdc_id:
                neighbors[search_data.loc[i].Vehicle_ID] = np.stack([search_data.loc[i].Local_Y, search_data.loc[i].Local_X], axis=-1)
        # sort the agents by distance
        sorted_neighbors = sorted(neighbors.items(), key=lambda item: np.linalg.norm(item[1] - self.current_xyh[:2]))
        sorted_neighbors = [obj for obj in sorted_neighbors if np.linalg.norm(obj[1] - self.current_xyh[:2])<=self.max_neighbor_distance ]
        # add neighbor agents into the array
        added_num = 0
        for neighbor in sorted_neighbors:
            neighbor_id = neighbor[0]
            neighbor_data = self.data[self.data["Vehicle_ID"]==neighbor_id][self.data["Global_Time"]<=current_time][self.data["Global_Time"]>=start_time].reset_index(inplace=False)
            neighbor_data = self.cal_heading(neighbor_data)
            if len(neighbor_data) < self.hist_len:
                continue
            if abs(neighbor_data.loc[len(neighbor_data)-1].Lane_ID - ego_lane) > 1:
                continue
            self.neighbors_id[added_num] = neighbor_id
            for i in range(0,self.hist_len):
                neighbors_states[added_num, i] = np.array([neighbor_data.loc[i].Local_Y, neighbor_data.loc[i].Local_X, neighbor_data.loc[i].Heading, neighbor_data.loc[i].velocity_y, 
                                                           neighbor_data.loc[i].velocity_x, neighbor_data.loc[i].acc_y,neighbor_data.loc[i].acc_x])
            added_num += 1
            # only consider 'num_neihgbors' agents
            if added_num >= self.num_neighbors:
                break
        # size = (self.neighbor_num, 20, 7)
        return neighbors_states.astype(np.float32), self.neighbors_id
    
    def build_map(self):
        '''
        map dictionary
        '''
        map_dict = {}
        ## lane1 - lane6
        for lane_num in range(1,7):
            lane = self.data[self.data["Lane_ID"]==lane_num].reset_index(inplace = False)
            center_x = round(np.mean(lane["Local_X"]),2)
            min_y, max_y = math.floor(lane["Local_Y"].min()), math.ceil(lane["Local_Y"].max())
            if lane_num <= 5:
                map_dict[lane_num] = np.array([[i, center_x, 0, 0, 30] for i in range(min_y,max_y,1)])
            else:
                map_dict[lane_num] = np.array([[i, center_x, 0, 1, 30] for i in range(min_y,max_y,1)])
        return map_dict
    
    def lane_map_feature(self, lane_id, pos):
        '''
        get_agents_map_feature
        '''
        lanes = [1,2,3,4,5,6]
        if lane_id not in lanes:
            return np.zeros((self.map_length,5))
        for i,point in enumerate(self.map_dict[lane_id]):
            if point[0]>pos:
                break
        if self.map_dict[lane_id].shape[0] - i > self.map_length:
            return self.map_dict[lane_id][i:i+self.map_length,:]
        else:
            map_feature = np.zeros((self.map_length,5))
            map_feature[0:self.map_dict[lane_id].shape[0] - i,:] = self.map_dict[lane_id][i:,:]
            return map_feature
        
    def ego_map_process(self,sdc_id, group, timestep):
        ego_map_feature = np.zeros((1, 3, self.map_length, 5))
        lane_id = group.loc[timestep].Lane_ID
        ego_pos = math.ceil(group.loc[timestep].Local_Y)
        #lane features
        lane_left = self.lane_map_feature(lane_id-1,ego_pos) 
        lane_ego = self.lane_map_feature(lane_id, ego_pos) 
        lane_right = self.lane_map_feature(lane_id+1, ego_pos) 
        ego_map_feature[0,...] = np.stack((lane_left,lane_ego,lane_right),axis=0)
        # size = (1, 3, 50, 5)
        return ego_map_feature.astype(np.float32)
    
    def neighbor_map_process(self, group, timestep):
        current_time = group.loc[timestep].Global_Time
        neighbor_map_feature = np.zeros((self.num_neighbors,3,self.map_length,5))
        for i, veh in enumerate(self.neighbors_id):
            lane_id = self.data[self.data["Vehicle_ID"]==self.neighbors_id[veh]][self.data["Global_Time"]==current_time].reset_index(inplace=False)["Lane_ID"][0]
            pos = math.ceil(self.data[self.data["Vehicle_ID"]==self.neighbors_id[veh]][self.data["Global_Time"]==current_time].reset_index(inplace=False)["Local_Y"][0])
            #lane features
            lane_left = self.lane_map_feature(lane_id-1,pos) 
            lane_ego = self.lane_map_feature(lane_id, pos) 
            lane_right = self.lane_map_feature(lane_id+1, pos)
            neighbor_map_feature[i,...] = np.stack((lane_left,lane_ego,lane_right),axis=0)
        # size = (self.neighbors_num, 3, 50, 5)
        return neighbor_map_feature.astype(np.float32)
    
    def groundtruth_process(self,sdc_id, group, timestep, neighbors):
        ground_truth = np.zeros(shape=(1+self.num_neighbors, self.future_len+1, 7))
        end_time = group.loc[timestep+self.future_len].Global_Time
        current_time = group.loc[timestep].Global_Time
        track_states = group.loc[timestep:timestep+self.future_len+1]
        ## ego ground_truth
        for i in range(timestep,timestep+self.future_len+1):
            ground_truth[0, i-timestep] = np.stack([track_states.loc[i].Local_Y, track_states.loc[i].Local_X, track_states.loc[i].Heading, track_states.loc[i].velocity_y, 
                                           track_states.loc[i].velocity_x, track_states.loc[i].acc_y,track_states.loc[i].acc_x], axis=-1)
        for i, id in enumerate(self.neighbors_id):
            track_states = self.data[self.data["Vehicle_ID"]==self.neighbors_id[id]][self.data["Global_Time"]>=current_time][self.data["Global_Time"]<=end_time].reset_index(inplace=False)   
            if len(track_states) < self.future_len+1:
                neighbors[id] = np.zeros((self.hist_len,7))
                continue
            track_states = self.cal_heading(track_states)
            for j in range(0, self.future_len+1):
                ground_truth[id+1, j] = np.stack([track_states.loc[j].Local_Y, track_states.loc[j].Local_X, track_states.loc[j].Heading, track_states.loc[j].velocity_y, 
                                               track_states.loc[j].velocity_x, track_states.loc[j].acc_y,track_states.loc[j].acc_x], axis=-1)
        self.gt = ground_truth
        #size = (1+self.neighbor_nums, futrue_lens + 1, 7)
        return ground_truth.astype(np.float32), neighbors.astype(np.float32)
    
    def normalize_data(self, ego, neighbors, ego_map, neighbors_map, ground_truth):
        # get the center and heading (local view)
        center, angle = self.current_xyh[:2], self.current_xyh[2]
        # normalize agents
        ego = agent_norm(ego, center, 0)
        ground_truth[0] = agent_norm(ground_truth[0], center, 0) 
        
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i] = agent_norm(neighbors[i], center, 0, impute=True)
                ground_truth[i+1] = agent_norm(ground_truth[i+1], center, 0)  
        # normalize map
        ego_map = map_norm(ego_map, center, 0)
        for i in range(neighbors.shape[0]):
            if neighbors_map[i, 1, -1, 0] != 0:
                neighbors_map[i] = map_norm(neighbors_map[i].reshape(1,3,self.map_length,5), center, 0)
        return ego.astype(np.float32), neighbors.astype(np.float32), ground_truth.astype(np.float32), ego_map.astype(np.float32), neighbors_map.astype(np.float32)
    
    def ego_plan(self):
        '''compute ego's control input in open loop'''
        ego_gt = self.gt[0,...]
        heading_change = [ego_gt[i+1,2] - ego_gt[i,2] for i in range(0,ego_gt.shape[0]-1)]
        steer = [heading_change[i]*4*10/ego_gt[i,3] for i in range(len(heading_change))]
        ego_control = np.zeros((2,self.future_len))
        ego_control[0,:] = ego_gt[:self.future_len,5]
        ego_control[1,:] = np.stack(steer,axis=-1)
        self.ego_command = ego_control
        return self.ego_command.astype(np.float32)
    
    def nan_check(self, ego, neighbors, ego_map, neighbors_map, ground_truth, ego_plan):
        if np.isnan(ego).any() or np.isnan(ego_map).any() or np.isnan(neighbors).any() or np.isnan(neighbors_map).any() or np.isnan(ground_truth).any() or np.isnan(ego_plan).any():
            return True
        else:
            return False
    
    def yaw_rate_check(self,agents):
        heading_change = np.array([agents[i,2] - agents[i-1,2] for i in range(1,len(agents))])
        if np.any(np.abs(heading_change) > 0.1):
            return True
        else:
            return False
        
    def process_data(self, save_path, viz = False):
        self.IDgroup_data = self.data.groupby(["Vehicle_ID","Total_Frames"])
        process_num = 0
        for name, group in self.IDgroup_data:
            sdc_id = name[0] #int
            frames = name[1]
            time_len = len(group) #int
            
            group = group.reset_index(inplace=False)
            # get_heading
            group = self.cal_heading(group)
            process_num += 1
            # start collect data
            for timestep in range(self.hist_len, time_len - self.future_len - 1, 10):
                ego = self.ego_process(group, timestep)
                # 检查主车yaw_rate是否越界，越界删去
                if self.yaw_rate_check(ego):
                    print("yaw rate out of bounds")
                    continue
                neighbors, _ = self.neighbors_process(sdc_id, group, timestep)
                # if no neighbors, then delete
                if neighbors[0,-1,0] == 0 and neighbors[1,-1,0] == 0 and neighbors[2,-1,0] == 0:
                    continue 
                
                ego_map = self.ego_map_process(sdc_id, group, timestep)
                neighbors_map = self.neighbor_map_process(group, timestep)
                ground_truth, neighbors = self.groundtruth_process(sdc_id, group, timestep, neighbors)
                # 检查真实值yaw_rate是否越界，越界删去
                if self.yaw_rate_check(ground_truth[0]) or self.yaw_rate_check(ground_truth[1]) or self.yaw_rate_check(ground_truth[2]) or self.yaw_rate_check(ground_truth[3]):
                    print("yaw rate out of bounds")
                    continue
                # 检查周围车yaw_rate是否越界，越界删去
                if self.yaw_rate_check(neighbors[0]) or self.yaw_rate_check(neighbors[1]) or self.yaw_rate_check(neighbors[2]):
                    print("yaw rate out of bounds")
                    continue
                ego_command = self.ego_plan()
                if np.any(np.abs(ego_command[0])>10):
                    print(timestep,"acceleration out of bounds")
                    continue
                if np.any(np.abs(ego_command[1])>1):
                    print(timestep,"angle out of bounds")
                    continue
                ego, neighbors, ground_truth, ego_map, neighbors_map = self.normalize_data(ego, neighbors, ego_map, neighbors_map, ground_truth)
                return ego,  ground_truth
                # 检查是否存在nan
                if self.nan_check(ego, neighbors, ego_map, neighbors_map, ground_truth, ego_command):
                    continue
                if viz:
                    scenario_viz(ego, neighbors, ego_map, ground_truth, sdc_id, frames, timestep)
                else:
                    pass
                # save data
                filename = f"{save_path}/{sdc_id}_{frames}_{timestep}.npz"
                np.savez(filename, ego=ego, neighbors=neighbors, ego_map=ego_map, neighbors_map=neighbors_map, 
                         gt_future_states=ground_truth, ego_plan = ego_command)
                print(f"{sdc_id}_{frames}_{timestep} has done! Progress:{process_num}/{len(self.IDgroup_data)}")
                del ego, neighbors, ground_truth, ego_map, neighbors_map, ego_command
            gc.collect()
    
    
#%% main
if __name__ == '__main__':
    data1 = pd.read_csv(r'data//0805_0820_us101_smoothed_21_.csv')
    ngsim = DataProcess(data1)
    ngsim.process_data("A_processed_data_805_820", False)

