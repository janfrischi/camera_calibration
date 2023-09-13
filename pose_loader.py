import os
import numpy as np
import matplotlib.pyplot as plt
import cam_params
from scipy.spatial.transform import Rotation


import matplotlib
matplotlib.use("TkAgg")

def rotmat_to_RPY_Ros(matrix):
    # Convert the rotation matrix to a Rotation object
    rotation = Rotation.from_matrix(matrix)

    # Get Euler angles in ZYX convention
    yaw, pitch, roll = rotation.as_euler('ZYX', degrees=False)

    return roll, pitch, yaw

def plot_camera(ax, R, T, cam_idx, scale=0.1, add_labels=False):
    # Create a 3x4 projection matrix
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = T

    # Define camera points in camera coordinate system
    points = np.array([
        [0, 0, 0, 1],
        [scale, 0, 0, 1],
        [0, scale, 0, 1],
        [0, 0, scale, 1]
    ]).T

    # Transform points to the world coordinate system
    world_points = np.linalg.inv(P) @ points
    world_points /= world_points[3, :]

    # Plot camera position and orientation
    ax.scatter(world_points[0, 0], world_points[1, 0], world_points[2, 0], marker="o", label=f"Camera {cam_idx}")
    ax.plot(world_points[0, :2], world_points[1, :2], world_points[2, :2], color="r", label="X" if add_labels else None)
    ax.plot(world_points[0, :3:2], world_points[1, :3:2], world_points[2, :3:2], color="g", label="Y" if add_labels else None)
    ax.plot(world_points[0, :4:3], world_points[1, :4:3], world_points[2, :4:3], color="b", label="Z" if add_labels else None)
    ax.axis('equal')

def plot_frame(ax, R, T, cam_idx, scale=0.1, add_labels=False):
    # Create a 3x4 projection matrix
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = T

    # Define camera points in camera coordinate system
    points = np.array([
        [0, 0, 0, 1],
        [scale, 0, 0, 1],
        [0, scale, 0, 1],
        [0, 0, scale, 1]
    ]).T

    # Transform points to the world coordinate system
    world_points = P @ points
    world_points /= world_points[3, :]

    # Plot camera position and orientation
    ax.scatter(world_points[0, 0], world_points[1, 0], world_points[2, 0], marker="o", label=f"Camera {cam_idx}")
    ax.plot(world_points[0, :2], world_points[1, :2], world_points[2, :2], color="r", label="X" if add_labels else None)
    ax.plot(world_points[0, :3:2], world_points[1, :3:2], world_points[2, :3:2], color="g", label="Y" if add_labels else None)
    ax.plot(world_points[0, :4:3], world_points[1, :4:3], world_points[2, :4:3], color="b", label="Z" if add_labels else None)
    ax.axis('equal')

info_folder = "/home/sopho/Videos/Rec_4/calib/info"  # Replace with the path to your info folder

# Load extrinsics
extrinsics = cam_params.load_extrinsics(info_folder)
print(extrinsics)
# Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

first_camera = True
for cam_idx, extrinsic_data in extrinsics.items():
    (R_l, T_l), (R_r, T_r) = extrinsic_data

    ##-------------------PRINT_CAM_FRAME---------------------------------
    # plot_camera(ax, R_l, T_l, cam_idx, add_labels=first_camera)
    # plot_camera(ax, R_r, T_r, cam_idx, add_labels=False)
    # first_camera = False
    ##-------------------PRINT_CAM_FRAME---------------------------------

# Set plot labels and show the plot
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="3d")

(R_l_1, T_l_1), (R_r_1, T_r_1) = extrinsics["1"]
(R_l_0, T_l_0), (R_r_0, T_r_0) = extrinsics["0"]

#define the coordinate frame transform
Trans_0_l, Trans_0_r = np.eye(4), np.eye(4)
Trans_0_l[:3, :3], Trans_0_r[:3, :3]  = R_l_0, R_r_0
Trans_0_l[:3, 3], Trans_0_r[:3, 3] = T_l_0, T_r_0
Trans_0_l, Trans_0_r = np.linalg.inv(Trans_0_l), np.linalg.inv(Trans_0_r)

Trans_1_l, Trans_1_r = np.eye(4), np.eye(4)
Trans_1_l[:3, :3], Trans_1_r[:3, :3]  = R_l_1, R_r_1
Trans_1_l[:3, 3], Trans_1_r[:3, 3] = T_l_1, T_r_1
Trans_1_l, Trans_1_r = np.linalg.inv(Trans_1_l), np.linalg.inv(Trans_1_r)

#define transform into ros frame
R_ros = np.eye(3)
R_ros[0,:]=[0,0,1]
R_ros[1,:]=[-1,0,0]
R_ros[2,:]=[0,-1,0]
R_ros= np.linalg.inv(R_ros)
T_ros=np.zeros(3)
Trans_ros = np.eye(4)
Trans_ros[:3, :3] = R_ros
Trans_ros[:3, 3] = T_ros

#negate Trans_ros for testing
# Trans_ros = np.eye(4)

#transform into ros coordinate frame

Trans_0_l, Trans_0_r = Trans_0_l @ Trans_ros, Trans_0_r @ Trans_ros
Trans_1_l, Trans_1_r = Trans_1_l @ Trans_ros, Trans_1_r @ Trans_ros

left_lens2base_link_t = np.array([0.01, -0.06, -0.015])
Trans_left2base = np.eye(4)
Trans_left2base[:3,3] = left_lens2base_link_t

base_0 = Trans_0_l.copy()
base_0 = base_0 @ Trans_left2base

base_1 = Trans_1_l.copy()
base_1 = base_1 @ Trans_left2base

# Test_matrix = np.linalg.inv(Trans_0_l) @ Trans_0_r
# Test_matrix = np.linalg.inv(Trans_1_l) @ Trans_1_r

# print(Test_matrix)


plot_frame(ax2, Trans_0_l[:3, :3], Trans_0_l[:3, 3],11, add_labels=False)
plot_frame(ax2, Trans_0_r[:3, :3], Trans_0_r[:3, 3],12, add_labels=False)
plot_frame(ax2, Trans_1_l[:3, :3], Trans_1_l[:3, 3],21, add_labels=False)
plot_frame(ax2, Trans_1_r[:3, :3], Trans_1_r[:3, 3],22, add_labels=False)
plot_frame(ax2, base_0[:3, :3], base_0[:3, 3],31, add_labels=False)
plot_frame(ax2, base_1[:3, :3], base_1[:3, 3],32, add_labels=True)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()


Trans_A_B = np.linalg.inv(base_0) @ base_1
r,p,y = rotmat_to_RPY_Ros(Trans_A_B[:3,:3])

print(f"The translation from base_link_A to base_link_B is: \n {Trans_A_B[:3,3]}")

print(f"The RPY angles from base_link_A to base_link_B is: \n {r,p,y}")

plt.show()


