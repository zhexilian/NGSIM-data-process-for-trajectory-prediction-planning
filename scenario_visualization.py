# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:14:37 2024

@author: 15834
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def scenario_viz(ego, neighbors, ego_map, ground_truth, id, frame, timestep):
    plt.figure(figsize=(18,8))
    plt.rcParams["axes.facecolor"] = "#f1f1f1"
    # visulization
    rect = plt.Rectangle((ego[-1, 0]-2, ego[-1, 1]-1), 4, 2, linewidth=2, color='#346888', alpha=0.4, zorder=3, label="ego_vehicle",
                        transform=mpl.transforms.Affine2D().rotate_around(*(ego[-1, 0], ego[-1, 1]), ego[-1, 2]) + plt.gca().transData)
    plt.gca().add_patch(rect)
    
    future = ground_truth[0][ground_truth[0][:, 0] != 0]
    plt.plot(future[:, 0], future[:, 1], '#a93245', marker='o', linewidth=3, alpha = 0.6, zorder=3, label="Future trajectory")
    plt.plot(ego[:, 0], ego[:, 1], '#275fab', marker='o', linewidth=3, alpha = 0.6, zorder=3, label="History trajectory")
    
    if_label = False
    for i in range(neighbors.shape[0]):
        if neighbors[i, -1, 0] != 0:
            if if_label:
                rect = plt.Rectangle((neighbors[i, -1, 0]-2, neighbors[i, -1, 1]-1), 
                                      4, 2, linewidth=2, color='#de425b', alpha=0.4, zorder=3,
                                      transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                future = ground_truth[i+1][ground_truth[i+1][:, 0] != 0]
                plt.plot(future[:, 0], future[:, 1], '#a93245', marker='>', linewidth=3, zorder=3, alpha = 0.6)
                plt.plot(neighbors[i, :, 0], neighbors[i, :, 1], '#275fab', marker='>', linewidth=3, zorder=3, alpha = 0.6)
                
            else:
                rect = plt.Rectangle((neighbors[i, -1, 0]-2, neighbors[i, -1, 1]-1), 
                                      4, 2, linewidth=2, color='#de425b', alpha=0.4, zorder=3, label = "surrounding vehicles",
                                      transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                future = ground_truth[i+1][ground_truth[i+1][:, 0] != 0]
                plt.plot(future[:, 0], future[:, 1], '#a93245', marker='>', linewidth=3, zorder=3, alpha = 0.6)
                plt.plot(neighbors[i, :, 0], neighbors[i, :, 1], '#275fab', marker='>', linewidth=3, zorder=3, alpha = 0.6)
                if_label = True
    
    line = np.linspace(-80, 80,100)
    plt.plot(line, [1.75]*len(line), '--', color = 'black', lw = 2)
    plt.plot(line, [-1.75]*len(line), '--', color = 'black', lw = 2)
    
    plt.ylim(-5.5,5.5)
    plt.xlim(-80,80)
    plt.xlabel("longitudinal in ego coordinate (m)",fontsize = 18)
    plt.ylabel("lateral in ego\n coordinate (m)",fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal')
    plt.tight_layout() 
    plt.legend(fontsize=12)
    plt.title(f"scenario:{id}_{frame}_{timestep}",fontsize=16)
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

#%%
if __name__ == '__main__':
    data = np.load("F:/23autumn/cherry/论文/offline_training/A_processed_data_750_805/1851_524_210.npz")
    scenario_viz(data['ego'], data['neighbors'], data['ego_map'], data['gt_future_states'], 1851,524, 210)
    # ego = data["ego"]
    # neighbors = data['neighbors']
    # gt = data['gt_future_states']
