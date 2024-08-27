# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:42:36 2024

@author: 15834
"""
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

def imputer(traj):
    x, y, v_x, v_y, theta, acc_x, acc_y = traj[:, 0], traj[:, 1], traj[:, 3], traj[:, 4], traj[:, 2], traj[:, 5], traj[:, 6]

    if np.any(x==0):
        for i in reversed(range(traj.shape[0])):
            if x[i] == 0:
                v_x[i] = v_x[i+1]
                v_y[i] = v_y[i+1]
                x[i] = x[i+1] - v_x[i]*0.1
                y[i] = y[i+1] - v_y[i]*0.1
                theta[i] = theta[i+1]
        return np.column_stack((x, y, theta, v_x, v_y, acc_x, acc_y))
    else:
        return np.column_stack((x, y, theta, v_x, v_y, acc_x, acc_y))
## 周围车辆转化自车坐标系
def agent_norm(traj, center, angle, impute=False):
    if impute:
        traj = imputer(traj)

    line = LineString(traj[:, :2])
    line_offset = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
    line_rotate = rotate(line_offset, -angle, origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[traj[:, :2]==0] = 0
    heading = wrap_to_pi(traj[:, 2] - angle)
    heading[traj[:, 2]==0] = 0

    velocity_x = traj[:, 3] * np.cos(angle) + traj[:, 4] * np.sin(angle)
    velocity_x[traj[:, 3]==0] = 0
    velocity_y = traj[:, 4] * np.cos(angle) - traj[:, 3] * np.sin(angle)
    velocity_y[traj[:, 4]==0] = 0
    acc_x = traj[:, 5] * np.cos(angle) + traj[:, 5] * np.sin(angle)
    acc_x[traj[:, 5]==0] = 0
    acc_y = traj[:, 6] * np.cos(angle) - traj[:, 6] * np.sin(angle)
    acc_y[traj[:, 6]==0] = 0
    return np.column_stack((line_rotate, heading, velocity_x, velocity_y, acc_x, acc_y))

def map_norm(map_line, center, angle):
    self_line = LineString(map_line[0,1,:,0:2])
    self_line = affine_transform(self_line, [1, 0, 0, 1, -center[0], -center[1]])
    self_line = rotate(self_line, -angle, origin=(0, 0), use_radians=True)
    self_line = np.array(self_line.coords)
    self_line[map_line[0,1,:,0:2]==0] = 0
    self_heading = wrap_to_pi(map_line[0,1,:,2] - angle)
    map_line[0,1,:,0:2] = self_line
    map_line[0,1,:,2] = self_heading
    
    if map_line[0, 0, :, 0][-1] == 0:
        pass
    else:
        left_line = LineString(map_line[0, 0, :, 0:2])
        left_line = affine_transform(left_line, [1, 0, 0, 1, -center[0], -center[1]])
        left_line = rotate(left_line, -angle, origin=(0, 0), use_radians=True)
        left_line = np.array(left_line.coords)
        left_line[map_line[0, 0, :, 0:2]==0] = 0
        left_heading = wrap_to_pi(map_line[0, 0, :, 2] - angle)
        left_heading[map_line[0, 0, :, 2]==0] = 0
        map_line[0,0,:,0:2] = left_line
        map_line[0,0,:,2] = left_heading
    
    if map_line[0, 2, :, 0][-1] == 0:
        pass
    else:
        right_line = LineString(map_line[0, 2, :, 0:2])
        right_line = affine_transform(right_line, [1, 0, 0, 1, -center[0], -center[1]])
        right_line = rotate(right_line, -angle, origin=(0, 0), use_radians=True)
        right_line = np.array(right_line.coords)
        right_line[map_line[0, 2, :, 0:2]==0] = 0
        right_heading = wrap_to_pi(map_line[0, 2, :, 2] - angle)
        right_heading[map_line[0, 2, :, 2]==0] = 0
        map_line[0,2,:,0:2] = right_line
        map_line[0,2,:,2] = right_heading
    
    return map_line
        
    
    
                                         

    