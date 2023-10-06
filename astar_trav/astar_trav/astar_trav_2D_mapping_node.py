import os
import cv2
import math
import time
import pickle
import numpy as np
import tensorflow as tf
import transforms3d
import copy
import traceback
import threading
from typing import Tuple, List, Dict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import rclpy
import rclpy.node
import rclpy.callback_groups
import rclpy.executors
from rclpy.executors import _rclpy
import rclpy.handle
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import astar_trav_msgs.msg as astar_trav_msgs
import builtin_interfaces.msg as builtin_interfaces
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs

from auto_scene_gen_core.vehicle_node import AutoSceneGenVehicleNode, VehicleNodeException, spin_vehicle_node
import auto_scene_gen_core.stamp_utils as stamp_utils
import auto_scene_gen_core.ros_image_converter as ros_img_converter

import astar_trav.video_utils as video_utils
import astar_trav.semseg_networks as semseg_networks
from astar_trav.semantic_segmentation import SemanticSegmentation
from astar_trav.astar_voting_path_planner import AstarVotingPathPlanner
from astar_trav.simple_path_follower_node import PathFollowerTrackingSnapshot
from astar_trav.astar_trav_2D_all_in_one_node import AstarImageSnapshot, AstarMapSnapshot, AstarPathSnapshot, AstarSnapshotsSummary

TRAV = 0
NON_TRAV = 1

def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))

def linear_interp(x1, x2, r):
    return x1 + r * (x2 - x1)


class AstarTrav2DMapper(AutoSceneGenVehicleNode):
    PLOT_PADDING = 5.
    
    def __init__(self, 
                node_name: str, 
                semseg_dict: Dict,
                path_planner_dict: Dict,
                height_thresh: float,                   
                camera_hfov: float,                                      # Camera horizontal FOV [rad]
                veh2cam_transform: Dict,
                veh2loc_transform: Dict,
                initial_sensor_proc_delay: int,                          # Begin processing sensor data for this long before creating the path planner timer
                b_use_vectorized_obstacle_vote_update: bool,
                nav_destination_topic_suffix: str,
                localization_topic_suffix: str,
                color_cam_topic_suffix: str,
                depth_cam_topic_suffix: str,
                map_state_topic_suffix: str,
                expected_camera_frame_rate: float,                      # Expected camera frame rtae in [Hz]
                ):
        super().__init__(node_name)

        self.semseg = SemanticSegmentation(**semseg_dict)
        self.image_size = semseg_dict['image_size'] # (H,W,C)
        self.num_pixels = self.image_size[0] * self.image_size[1]
        self.image_aspect_ratio = self.image_size[1] / self.image_size[0]
        self.expected_camera_frame_rate = expected_camera_frame_rate

        self.height_thresh = height_thresh
        self.path_planner = AstarVotingPathPlanner(**path_planner_dict)
        self.landscape_size = path_planner_dict['env_size']

        # Note: transforms3d.*2mat() returns the rotation matrix from body to base frame
        self.veh2cam_offset = veh2cam_transform['offset'].reshape(3,1)
        self.veh2cam_orientation = veh2cam_transform['orientation']
        self.R_cam2veh = transforms3d.euler.euler2mat(self.veh2cam_orientation[0], self.veh2cam_orientation[1], self.veh2cam_orientation[2], 'rzyx')

        self.veh2loc_offset = veh2loc_transform['offset'].reshape(3,1)
        self.veh2loc_orientation = veh2loc_transform['orientation']
        self.R_loc2veh = transforms3d.euler.euler2mat(self.veh2loc_orientation[0], self.veh2loc_orientation[1], self.veh2loc_orientation[2], 'rzyx')

        # What happens if img height and width are different?
        self.camera_hfov = camera_hfov
        self.camera_vfov = 2. * math.atan((math.tan(self.camera_hfov/2.) / self.image_aspect_ratio))
        self.rad_per_xpixel = self.camera_hfov / self.image_size[1]
        self.rad_per_ypixel = self.camera_vfov / self.image_size[0]

        # Create mask to element-wise multiply depth image to create xyz positions in the camera frame
        # pixel_yaw and pixel_pitch correspond to the yaw/pitch angles a laser pointer would have to orient to w.r.t the sensor's line of sight to point at a particular pixel
        # These yaw/pitch angles are w.r.t to the sensor frame coordinates (x = fwd, y = left, z = up), NOT the image frame cordinates
        self.cam_xyzmask = np.ones(self.image_size, dtype=np.float32)
        for i in range(self.image_size[0]):
            pixel_pitch = -0.5 * self.camera_hfov + self.rad_per_ypixel * (i + 0.5) # pixel_pitch increases top to bottom across the image
            for j in range(self.image_size[1]):
                pixel_yaw = 0.5 * self.camera_hfov - self.rad_per_xpixel * (j + 0.5) # pixel_yaw decreases left to right across the image
                self.cam_xyzmask[i,j,1] = math.tan(pixel_yaw) # y channel mask
                self.cam_xyzmask[i,j,2] = -math.tan(pixel_pitch) # z channel mask

        # Current sensor data
        self.color_img_data = None
        self.prev_color_img_data = None
        self.depth_img_data = None
        self.prev_depth_img_data = None
        self.img_stamp = None
        self.vehicle_position = None # Vehicle position
        self.vehicle_orientation = None # Vehicle orientation

        self.goal_point = None
        self.initial_sensor_proc_delay = initial_sensor_proc_delay
        self.b_use_vectorized_obstacle_vote_update = b_use_vectorized_obstacle_vote_update
        self.initial_cam_proc_stamp = None
        self.initial_astar_replan_stamp = None

        self.max_votes_per_vertex = path_planner_dict['max_votes_per_vertex']

        # Variables for saving data
        self.image_snapshots : List[AstarImageSnapshot] = []
        self.map_snapshots : List[AstarMapSnapshot] = []
        self.snapshots_summary = AstarSnapshotsSummary()

        sensor_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Subs
        self.nav_topic = self.worker_topic_prefix + nav_destination_topic_suffix
        self.loc_topic = self.vehicle_topic_prefix + localization_topic_suffix
        self.color_topic = self.vehicle_topic_prefix + color_cam_topic_suffix
        self.depth_topic = self.vehicle_topic_prefix + depth_cam_topic_suffix
        self.nav_destination_sub = self.create_subscription(geometry_msgs.Pose, self.nav_topic, self.nav_destination_cb, 10)
        self.localization_sub = self.create_subscription(geometry_msgs.PoseStamped, self.loc_topic, self.localization_cb, 10)
        self.color_cam_sub = self.create_subscription(sensor_msgs.Image, self.color_topic, self.color_cam_cb, 10)
        self.depth_cam_sub = self.create_subscription(sensor_msgs.Image, self.depth_topic, self.depth_cam_cb, 10)

        # Pubs
        self.map_state_topic = self.vehicle_topic_prefix + map_state_topic_suffix
        self.map_state_pub = self.create_publisher(astar_trav_msgs.MapState2D, self.map_state_topic, 10)

        self.log("info", f'Initialized A* trav mapping node for worker ID {self.wid}.')

        # Debug variables, used in logging
        self.num_color = 0
        self.num_depth = 0
        self.image_proc_time = 0.
        self.max_proc_stamp = None
        self.max_proc_time = 0.
    
    def reset(self):
        self.path_planner.clear_map()
        self.goal_point = None

        self.color_img_data = None
        self.prev_color_img_data = None
        self.depth_img_data = None
        self.prev_depth_img_data = None
        self.img_stamp = None

        self.vehicle_position = None
        self.vehicle_orientation = None
        self.R_veh2w = None
        self.initial_cam_proc_stamp = None
        self.initial_astar_replan_stamp = None

        self.image_snapshots.clear()
        self.map_snapshots.clear()
        self.snapshots_summary = AstarSnapshotsSummary()

        self.num_color = 0
        self.num_depth = 0
        self.image_proc_time = 0.
        self.max_proc_stamp = None
        self.max_proc_time = 0.
        self.log("info", f"2D mapper has been reset.")

    def nav_destination_cb(self, msg: geometry_msgs.Pose):
        if not self.vehicle_ok():
            return
    
        self.goal_point = self.path_planner.get_nearest_vertex(np.array([msg.position.x, msg.position.y]), b_major=True)
    
    def localization_cb(self, msg: geometry_msgs.PoseStamped):
        if not self.vehicle_ok():
            return
        
        self.vehicle_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32).reshape(3,1) - self.veh2loc_offset
        self.vehicle_orientation = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] # Let's assume aligned with vehicle frame
        self.R_veh2w = transforms3d.quaternions.quat2mat(self.vehicle_orientation) # Rotation matrix body to world
        self.euler = transforms3d.euler.quat2euler(self.vehicle_orientation, 'rzyx')

    def color_cam_cb(self, msg: sensor_msgs.Image):
        if not self.vehicle_ok():
            return

        self.num_color += 1
        self.prev_color_img_data = self.color_img_data
        self.color_img_data = (msg.header.stamp, ros_img_converter.image_msg_to_numpy(msg, keep_as_3d_tensor=True))
        self.process_sensor_data()
    
    def depth_cam_cb(self, msg: sensor_msgs.Image):
        if not self.vehicle_ok():
            return
        
        self.num_depth += 1
        self.prev_depth_img_data = self.depth_img_data
        self.depth_img_data = (msg.header.stamp, ros_img_converter.image_msg_to_numpy(msg, keep_as_3d_tensor=True))
        self.process_sensor_data()

    def process_sensor_data(self):
        """Process camera sensor data. Align color/depth camera data based on time stamps and update internal map.
        Can only be called from one of the camera callbacks.
        """
        if not self.vehicle_ok() or self.prev_color_img_data is None or self.prev_depth_img_data is None or self.vehicle_position is None:
            return
        
        color_img = None
        depth_img = None
        img_stamp = None

        # Find most recent color and depth images with matching header sequences
        if stamp_utils.do_ros_stamps_match(self.color_img_data[0], self.depth_img_data[0]):
            color_img = self.color_img_data[1]
            depth_img = self.depth_img_data[1]
            img_stamp = self.color_img_data[0]

        elif stamp_utils.do_ros_stamps_match(self.color_img_data[0], self.prev_depth_img_data[0]):
            color_img = self.color_img_data[1]
            depth_img = self.prev_depth_img_data[1]
            img_stamp = self.color_img_data[0]

        elif stamp_utils.do_ros_stamps_match(self.prev_color_img_data[0], self.depth_img_data[0]):
            color_img = self.prev_color_img_data[1]
            depth_img = self.depth_img_data[1]
            img_stamp = self.prev_color_img_data[0]

        # Update map with latest info if new image stamp does not match the last stamp
        if color_img is not None and (self.img_stamp is None or not stamp_utils.do_ros_stamps_match(self.img_stamp, img_stamp)):
            self.img_stamp = img_stamp
            self.update_map(color_img, depth_img, img_stamp)

    def update_map(self, color_img: np.float32, depth_img: np.float32, img_stamp: builtin_interfaces.Time):
        """Update world map based on most recent camera data"""
        if not self.vehicle_ok() or self.vehicle_position is None:
            return

        start_time = time.time()
        if self.initial_cam_proc_stamp is None:
            self.initial_cam_proc_stamp = self.sim_clock.now().to_msg()

        # Create image snapshot
        img_snapshot = AstarImageSnapshot()
        img_snapshot.raw_color_img = color_img # 3D numpy array
        img_snapshot.raw_depth_img  = depth_img # 3D numpy array
        img_snapshot.stamp = img_stamp
        
        # Convert depth data to xyz coordinates in world frame
        xyz_cam = (self.cam_xyzmask * depth_img).reshape(self.num_pixels, 3).T # Each column is an (x,y,z) vector expressed in the cam frame
        xyz_veh = self.R_cam2veh.dot(xyz_cam) + self.veh2cam_offset
        xyz_world = self.R_veh2w.dot(xyz_veh) + self.vehicle_position # Each column is an (x,y,z) vector expressed in the world frame

        # Compute mask labels for color image
        probs, mask_labels = self.semseg.generate_predictions(color_img)
        img_snapshot.predicted_mask_labels = mask_labels
        self.image_snapshots.append(img_snapshot)
        mask_labels = mask_labels.copy().flatten()

        # Update path planner's map with nontraversable points that are beneath the height threshold
        if self.b_use_vectorized_obstacle_vote_update:
            # Vectorized method (MUCH MUCH faster)
            obs_idx = (mask_labels == NON_TRAV) & (xyz_world[2,:] <= self.height_thresh) # & is bitwise AND
            self.path_planner.add_obstacle_votes_from_array(xyz_world[0:2,obs_idx].T)
        else:
            # Inefficient for-loop approach
            for i in range(self.num_pixels):
                if mask_labels[i] == NON_TRAV and xyz_world[2,i] <= self.height_thresh:
                    self.path_planner.add_obstacle_vote(xyz_world[0:2, i]) # Add obstacle (x,y) location to map

        image_proc_duration = time.time()-start_time
        self.image_proc_time += image_proc_duration
        self.path_planner.update_compact_numpy_arrays()

        map_snapshot = AstarMapSnapshot()
        map_snapshot.vehicle_position = self.vehicle_position
        map_snapshot.vehicle_orientation = self.vehicle_orientation
        map_snapshot.obstacle_votes = self.path_planner.obstacle_votes.copy()
        map_snapshot.stamp = img_stamp
        # if len(self.map_snapshots) > 0:
        #     map_snapshot.opt_path = self.map_snapshots[-1].opt_path
        #     map_snapshot.opt_cost = self.map_snapshots[-1].opt_cost
        #     map_snapshot.direct_path = self.map_snapshots[-1].direct_path
        #     map_snapshot.direct_path_cost = self.map_snapshots[-1].direct_path_cost
        self.map_snapshots.append(map_snapshot)

        self.snapshots_summary.image_processing_duration.append(image_proc_duration)
        if image_proc_duration > self.max_proc_time:
            self.max_proc_time = image_proc_duration
            self.max_proc_stamp = img_stamp

        if stamp_utils.get_stamp_dt(self.initial_cam_proc_stamp, self.sim_clock.now().to_msg()) >= self.initial_sensor_proc_delay:
            map_msg = astar_trav_msgs.MapState2D()
            map_msg.header.stamp = img_stamp
            map_msg.header.frame_id = "/world"
            map_msg.env_size_x = self.path_planner.env_size[0]
            map_msg.env_size_y = self.path_planner.env_size[1]
            map_msg.border_size = self.path_planner.border_size
            map_msg.minor_cell_size = self.path_planner.minor_cell_size[0]
            map_msg.bell_curve_sigma = self.path_planner.bell_curve_sigma

            # Add obstacle info. If there are no votes, then we can leave these lists empty.
            if np.sum(self.path_planner.obstacle_votes):
                map_msg.obstacle_votes = self.path_planner.obstacle_votes.tolist()
                map_msg.obstacle_votes_compact = self.path_planner.obs_votes_compact_np.tolist()
                map_msg.obstacle_locations_compact_x = self.path_planner.obs_locations_compact_np[:,0].tolist()
                map_msg.obstacle_locations_compact_y = self.path_planner.obs_locations_compact_np[:,1].tolist()
            self.map_state_pub.publish(map_msg)

    def save_trav_semseg_video(self):
        """Make a video of the camera data and the predicted semantic segmentation images"""
        self.log("info", "Saving trav semseg video...")

        # Create semantic segmentation images
        for img_snapshot in self.image_snapshots:
            predicted_seg_img = self.semseg.generate_segmentation_image(img_snapshot.predicted_mask_labels)
            img_snapshot.predicted_seg_img = predicted_seg_img

        dt_list = []
        for i in range(0,len(self.image_snapshots)-1):
            dt_list.append(stamp_utils.get_stamp_dt(self.image_snapshots[i].stamp, self.image_snapshots[i+1].stamp))
        avg_dt = np.mean(dt_list)
        fps = 1. / avg_dt

        # Make movie of raw color image and predicted semseg image
        if self.b_use_ssh:
            save_dir = self.temp_save_dir
        else:
            save_dir = self.save_dir
        video_filename_avi = os.path.join(save_dir, f"trav_semseg.avi") # Uncompressed video
        video_filename_mp4 = os.path.join(save_dir, f"trav_semseg.mp4") # Compressed video
        fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G') # Use for .avi, no compression
        fourcc_mp4 = cv2.VideoWriter_fourcc('a','v','c','1') # H.264 codec, use for .mp4, compression, plays in vscode
        frame_size = (self.image_size[0]*2, self.image_size[1]) # Width x Height
        video_avi = cv2.VideoWriter(video_filename_avi, fourcc_avi, fps, frame_size)
        video_mp4 = cv2.VideoWriter(video_filename_mp4, fourcc_mp4, fps, frame_size)
        for img_snapshot in self.image_snapshots:
            raw_color_img = cv2.cvtColor(img_snapshot.raw_color_img, cv2.COLOR_RGB2BGR)
            predicted_seg_img = cv2.cvtColor(img_snapshot.predicted_seg_img, cv2.COLOR_RGB2BGR)
            frame = np.concatenate((raw_color_img, predicted_seg_img), axis=1) # Color on left, seg on right
            video_avi.write(frame)
            video_mp4.write(frame)
        video_avi.release()
        video_mp4.release()

        if self.b_use_ssh:
            remote_video_filename_avi = os.path.join(self.save_dir, f"trav_semseg.avi") # Uncompressed video
            remote_video_filename_mp4 = os.path.join(self.save_dir, f"trav_semseg.mp4") # Compressed video
            self.save_file_on_remote_asg_client(video_filename_avi, remote_video_filename_avi)
            self.save_file_on_remote_asg_client(video_filename_mp4, remote_video_filename_mp4)

        # NOTE: Saving this as a pickle consumes an enormous amount of space
        # pickle_filename = os.path.join(self.save_dir, 'astar_image_snapshots.pkl')
        # with open(pickle_filename, 'wb') as f:
        #     pickle.dump(self.image_snapshots, f, pickle.HIGHEST_PROTOCOL)

    def save_path_planner_video(self):
        """Make a video of the map updates along with the vehicle path and generated A* paths"""        
        self.log("info", "Waiting for tracking snapshots file...")
        tracking_snapshots : List[PathFollowerTrackingSnapshot] = self.load_pickle_object_from_asg_client("path_follower_snapshots.pkl")
        if tracking_snapshots is None:
            self.log("warn", f"Failed to load tracking snapshots. Cannot save path planner video.")
            return
        self.log("info", "Loaded tracking snapshots")
       
        self.log("info", "Waiting for path snapshots file...")
        path_snapshots : List[AstarPathSnapshot] = self.load_pickle_object_from_asg_client("astar_path_snapshots.pkl")
        if path_snapshots is None:
            self.log("warn", f"Failed to load A* path snapshots. Cannot save path planner video.")
            return
        self.log("info", "Loaded A* path snapshots")
        
        self.log("info", "Saving path planner video...")

        if self.b_use_ssh:
            save_dir = self.temp_save_dir
        else:
            save_dir = self.save_dir

        # Plot colors and zorders
        path_color = 'black'
        tracking_color = 'blue'
        gt_obs_color = 'forestgreen' # Forest green
        ssa_plot_color = {
            'ssa_tree': (0.133, 0.545, 0.133, 1.), # Forest green
            'ssa_bush': (0.647, 0.165, 0.165, 1.), # Brown RGB(165,42,42)
            'ssa_rock': (0.5, 0.5, 0.5, 1.) # Gray
        }
        ssa_name = {
            'ssa_tree': 'Tree',
            'ssa_bush': 'Bush',
            'ssa_rock': 'Rock'
        }
        SSA_PLOT_INFO = {
            'ssa_tree': {"color": "forestgreen", "marker": "^", "markersize": 8**2, "label": "Tree"}, # Forest green (0.133, 0.545, 0.133, 1.), tri_up
            'ssa_bush': {"color": "lime", "marker": "p", "markersize": 8**2, "label": "Bush"}, # Brown (0.647, 0.165, 0.165, 1.) RGB(165,42,42), pentagon
            'ssa_rock': {"color": (0.5, 0.5, 0.5, 1.), "marker": "o", "markersize": 8**2, "label": "Rock"},  # Gray, circle
        }
        obs_color = np.zeros((self.path_planner.obstacle_votes.size,4), dtype=np.float32)
        obs_color[:,0] = 1. # Red, with varying intensity
        obs_legend_color = (1.,0.,0.,1.)

        obs_vote_zorder = 1
        ssa_zorder = 2
        tracking_zorder = 5
        astar_zorder = 4
        path_zorder = 3

        video_frames_full = []
        video_frames_cropped = []
        video_frames_vehicle_centered = []
        
        # Fill in remaining map snapshot fields
        path_idx = 0
        track_idx = 0
        for i in range(len(self.map_snapshots)):
            map_snapshot = self.map_snapshots[i]
            map_snapshot.minor_vertices = self.path_planner.minor_vertices.copy()
            map_snapshot.max_votes_per_vertex = self.path_planner.max_votes_per_vertex

            # Get most recent values
            if i > 0:
                map_snapshot.opt_path = self.map_snapshots[i-1].opt_path
                map_snapshot.opt_cost = self.map_snapshots[i-1].opt_cost
                map_snapshot.direct_path = self.map_snapshots[i-1].direct_path
                map_snapshot.direct_path_cost = self.map_snapshots[i-1].direct_path_cost
                map_snapshot.approx_tracking_waypoint = self.map_snapshots[i-1].approx_tracking_waypoint

                while path_idx < len(path_snapshots) and stamp_utils.get_stamp_dt(path_snapshots[path_idx].stamp, map_snapshot.stamp) >= 0:
                    map_snapshot.opt_path = path_snapshots[path_idx].opt_path
                    map_snapshot.opt_cost = path_snapshots[path_idx].opt_path_cost
                    map_snapshot.direct_path = path_snapshots[path_idx].direct_path
                    map_snapshot.direct_path_cost = path_snapshots[path_idx].direct_path_cost
                    path_idx += 1

                while track_idx < len(tracking_snapshots) and stamp_utils.get_stamp_dt(tracking_snapshots[track_idx].stamp, map_snapshot.stamp) >= 0:
                    map_snapshot.approx_tracking_waypoint = tracking_snapshots[track_idx].tracking_waypoint
                    track_idx += 1
   
        start_location = None
        goal_location = self.map_snapshots[-1].opt_path[-1]

        # Path extremes: vehicle's path, and min/max values from opt path
        path_extremes_x = []
        path_extremes_y = []
        for j in range(len(self.map_snapshots)):
            # Add vehicle path
            path_extremes_x.append(self.map_snapshots[j].vehicle_position[0])
            path_extremes_y.append(self.map_snapshots[j].vehicle_position[1])

            # Add min/max positions from optimal path
            if self.map_snapshots[j].opt_path is not None:
                opt_path = np.array(self.map_snapshots[j].opt_path).T # Each col is an (x,y) pair
                path_extremes_x.append(np.min(opt_path[0,:]))
                path_extremes_x.append(np.max(opt_path[0,:]))
                path_extremes_y.append(np.min(opt_path[1,:]))
                path_extremes_y.append(np.max(opt_path[1,:]))

                if start_location is None:
                    start_location = self.map_snapshots[j].opt_path[0]

        # Make video of vehicle traveling and show path planner and path follower updates
        for i in range(len(self.map_snapshots)):
            snapshot = self.map_snapshots[i]
            fig, ax = plt.subplots(num=1)

            ax.scatter(start_location[0], start_location[1], s=8**2, zorder=1, color="red", marker='x', label="Start Location")
            ax.scatter(goal_location[0], goal_location[1], s=8**2, zorder=1, color="blue", marker='x', label="Goal Location")

            # Plot vehicle path
            path_x = []
            path_y = []
            for j in range(i+1):
                path_x.append(self.map_snapshots[j].vehicle_position[0])
                path_y.append(self.map_snapshots[j].vehicle_position[1])
            ax.plot(path_x, path_y, zorder=path_zorder, linewidth=1, color=path_color, label="Vehicle Path")
            # ax.scatter([snapshot.vehicle_position[0]], [snapshot.vehicle_position[1]], s=8, marker='o', zorder=path_zorder, color=path_color, label="Vehicle")
            
            # Plot triangle at vehicle position aligned with vehicle orientation. Default triangle is up, so subtract 90 deg.
            euler = transforms3d.euler.quat2euler(snapshot.vehicle_orientation, 'rzyx')
            ax.scatter([snapshot.vehicle_position[0]], [snapshot.vehicle_position[1]], s=6**2, marker=(3, 0, (euler[0]*180./math.pi) - 90.), zorder=path_zorder, color=path_color, label="Vehicle")
            
            # Plot A* path (ignore the first frame if it doesn't have any)
            if snapshot.opt_path is not None:
                opt_path = np.array(snapshot.opt_path)
                opt_path_x = opt_path[:,0].tolist()
                opt_path_y = opt_path[:,1].tolist()
                ax.plot(opt_path_x, opt_path_y, zorder=astar_zorder, linestyle='--', linewidth=1, color=path_color, label=f"A* Path\nCost: {snapshot.opt_cost:.2f}")

                direct_path = np.array(snapshot.direct_path)
                direct_path_x = direct_path[:,0].tolist()
                direct_path_y = direct_path[:,1].tolist()
                ax.plot(direct_path_x, direct_path_y, zorder=astar_zorder, linestyle='--', linewidth=1, color=path_color, alpha=0.5, label=f"Direct Path\nCost: {snapshot.direct_path_cost:.2f}")

            else:
                # Plot dummy point out of view so we can keep the legends the same throughout all frames
                ax.plot([-self.path_planner.env_size[0]], [-self.path_planner.env_size[1]], zorder=astar_zorder, linestyle='--', linewidth=1, color=path_color, label=f"A* Path\nCost: N/A")
                ax.plot([-self.path_planner.env_size[0]], [-self.path_planner.env_size[1]], zorder=astar_zorder, linestyle='--', linewidth=1, color=path_color, alpha=0.5, label=f"Direct Path\nCost: N/A")

            # Plot tracking point for current frame
            if snapshot.approx_tracking_waypoint is not None:
                ax.scatter([snapshot.approx_tracking_waypoint[0]], [snapshot.approx_tracking_waypoint[1]], s=8**2, marker='*', zorder=tracking_zorder, linewidths=0, color=tracking_color, label="Tracking Point")
            else:
                # Plot dummy point out of view so we can keep the legends the same throughout all frames
                ax.scatter([-self.path_planner.env_size[0]], [-self.path_planner.env_size[1]], s=8**2, marker='*', zorder=tracking_zorder, linewidths=0, color=tracking_color, label="Tracking Point")

            # Plot obstacle votes (use vectorized approach for faster performance)
            obs_x = snapshot.minor_vertices[:,0].tolist()
            obs_y = snapshot.minor_vertices[:,1].tolist()
            obs_color[:,3] = snapshot.obstacle_votes / snapshot.max_votes_per_vertex
            ax.scatter(obs_x, obs_y, s=8**2, marker='.', zorder=obs_vote_zorder, linewidths=0, c=obs_color) # Use 'c' arg for providing a map/list of colors
            ax.scatter([-self.path_planner.env_size[0]], [-self.path_planner.env_size[1]], s=8**2, marker='.', zorder=obs_vote_zorder, linewidths=0, color=obs_legend_color, label="Predicted Obstacle") # Add label to legend by plotting dummy point

            # Plot SSAs
            ssa_array = self.scene_description.ssa_array
            for ssa_type, ssa_type_info in SSA_PLOT_INFO.items():
                ssa_x = []
                ssa_y = []
                for i in range(len(ssa_array)):
                    if ssa_type in str.lower(ssa_array[i].path_name):
                        layout = ssa_array[i]
                        for j in range(layout.num_instances):
                            if layout.visible[j]:
                                ssa_x.append(layout.x[j])
                                ssa_y.append(layout.y[j])

                if len(ssa_x) > 0:
                    ax.scatter(ssa_x, ssa_y, s=ssa_type_info["markersize"], zorder=ssa_zorder, color=ssa_type_info["color"], marker=ssa_type_info["marker"], linewidths=0, label=ssa_type_info["label"])

            # ax.set_xlabel("X [m]")
            # ax.set_ylabel("Y [m]")
            # ax.set_aspect('equal', adjustable='box')
            # legend = ax.legend(loc='center left', bbox_to_anchor=(1.025, 0.5)) # Place legend outside plot

            fontsize=16
            ax.set_xlabel("X [m]", fontsize=fontsize)
            ax.set_ylabel("Y [m]", fontsize=fontsize)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.tick_params(axis="both", which="major", labelsize=fontsize)
            ax.set_aspect('equal', adjustable='box')
            legend = ax.legend(loc='center left', bbox_to_anchor=(1.025, 0.5), fontsize=12) # Place legend outside plot
            # handles, labels = ax.get_legend_handles_labels()

            # Plot entire scene
            xmin = min(min(path_extremes_x), 0.)
            ymin = min(min(path_extremes_y), 0.)
            xmax = max(max(path_extremes_x), self.landscape_size[0])
            ymax = max(max(path_extremes_y), self.landscape_size[1])
            ax.set_xlim([xmin - self.PLOT_PADDING, xmax + self.PLOT_PADDING])
            ax.set_ylim([ymin - self.PLOT_PADDING, ymax + self.PLOT_PADDING])
            fig.tight_layout(pad=0.5)
            # fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, 1), fontsize=10)
            file_name_full = os.path.join(save_dir, "path_planner_full_temp.png")
            fig.savefig(file_name_full, bbox_extra_artists=(legend,), bbox_inches='tight')
            # fig.savefig(file_name_full, bbox_inches='tight')

            # Plot cropped scene (just big enough to fit the full vehicle path)
            min_width = 20.
            xmin = min(min(path_extremes_x), self.goal_point[0])
            ymin = min(min(path_extremes_y), self.goal_point[1])
            xmax = max(max(path_extremes_x), self.goal_point[0])
            ymax = max(max(path_extremes_y), self.goal_point[1])
            if math.fabs(xmax - xmin) < min_width:
                mid = (xmin + xmax)/2.
                xmin = mid - min_width/2.
                xmax = mid + min_width/2.
            if math.fabs(ymax - ymin) < min_width:
                mid = (ymin + ymax)/2.
                ymin = mid - min_width/2.
                ymax = mid + min_width/2.
            ax.set_xlim([xmin - self.PLOT_PADDING, xmax + self.PLOT_PADDING])
            ax.set_ylim([ymin - self.PLOT_PADDING, ymax + self.PLOT_PADDING])
            fig.tight_layout(pad=0.5)
            # fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, 1), fontsize=10)
            file_name_cropped = os.path.join(save_dir, "path_planner_cropped_temp.png")
            fig.savefig(file_name_cropped, bbox_extra_artists=(legend,), bbox_inches='tight')
            # fig.savefig(file_name_cropped, bbox_inches='tight')

            # Plot cropped scene centered about the vehicle (vehicle stays in center of frame)
            # frame_width = 40.
            # ax.set_title("   ") # Set an empty title so that image size is consistent when following vehicle
            # ax.set_xlim([snapshot.vehicle_position[0] - frame_width/2., snapshot.vehicle_position[0] + frame_width/2.])
            # ax.set_ylim([snapshot.vehicle_position[1] - frame_width/2., snapshot.vehicle_position[1] + frame_width/2.])
            # file_name_vehicle_centered = os.path.join(save_dir, "path_planner_vehicle_centered_temp.png")
            # fig.savefig(file_name_vehicle_centered, bbox_extra_artists=(legend,), bbox_inches='tight')
            
            fig.clf()

            # Get images now, since we are saving them under the same name
            video_frames_full.append(cv2.imread(file_name_full))
            video_frames_cropped.append(cv2.imread(file_name_cropped))
            # video_frames_vehicle_centered.append(cv2.imread(file_name_vehicle_centered))

        if video_frames_full[0].shape != video_frames_full[-1].shape:
            start_idx = 0
            for i in range(len(video_frames_full)):
                if video_frames_full[i].shape == video_frames_full[-1].shape:
                    start_idx = i
                    break
            video_frames_full = video_frames_full[start_idx:]
            self.log("warn", f"Path planner full video frames have incosistent shapes: start frame is {video_frames_full[0].shape} and last frame is {video_frames_full[-1].shape}. Will start at frame {start_idx+1}.")
        
        if video_frames_cropped[0].shape != video_frames_cropped[-1].shape:
            start_idx = 0
            for i in range(len(video_frames_cropped)):
                if video_frames_cropped[i].shape == video_frames_cropped[-1].shape:
                    start_idx = i
                    break
            video_frames_cropped = video_frames_cropped[start_idx:]
            self.log("warn", f"Path planner cropped video frames have incosistent shapes: start frame is {video_frames_cropped[0].shape} and last frame is {video_frames_cropped[-1].shape}. Will start at frame {start_idx+1}.")
        
        # if video_frames_vehicle_centered[0].shape != video_frames_vehicle_centered[-1].shape:
        #     self.log("warn", f"Path planner vehicle-centered video frames have incosistent shapes: start frame is {video_frames_vehicle_centered[0].shape} and last frame is {video_frames_vehicle_centered[-1].shape}.")

        dt_list = []
        for i in range(0,len(self.map_snapshots)-1):
            dt_list.append(stamp_utils.get_stamp_dt(self.map_snapshots[i].stamp, self.map_snapshots[i+1].stamp))
        avg_dt = np.mean(dt_list)
        fps = 1. / avg_dt

        # Save video files
        frame_size = (video_frames_full[0].shape[1], video_frames_full[0].shape[0]) # Width x Height
        video_utils.write_video(os.path.join(save_dir, f"path_planner_full"), ".avi", fps, frame_size, video_frames_full)
        video_utils.write_video(os.path.join(save_dir, f"path_planner_full"), ".mp4", fps, frame_size, video_frames_full)
        
        frame_size = (video_frames_cropped[0].shape[1], video_frames_cropped[0].shape[0]) # Width x Height
        video_utils.write_video(os.path.join(save_dir, f"path_planner_cropped"), ".avi", fps, frame_size, video_frames_cropped)
        video_utils.write_video(os.path.join(save_dir, f"path_planner_cropped"), ".mp4", fps, frame_size, video_frames_cropped)
        
        # frame_size = (video_frames_vehicle_centered[0].shape[1], video_frames_vehicle_centered[0].shape[0]) # Width x Height
        # video_utils.write_video(os.path.join(save_dir, f"path_planner_vehicle_centered"), ".avi", fps, frame_size, video_frames_vehicle_centered)
        # video_utils.write_video(os.path.join(save_dir, f"path_planner_vehicle_centered"), ".mp4", fps, frame_size, video_frames_vehicle_centered)

        if self.b_use_ssh:
            self.save_file_on_remote_asg_client(os.path.join(save_dir, f"path_planner_full.avi"), os.path.join(self.save_dir, f"path_planner_full.avi"))
            self.save_file_on_remote_asg_client(os.path.join(save_dir, f"path_planner_full.mp4"), os.path.join(self.save_dir, f"path_planner_full.mp4"))

            self.save_file_on_remote_asg_client(os.path.join(save_dir, f"path_planner_cropped.avi"), os.path.join(self.save_dir, f"path_planner_cropped.avi"))
            self.save_file_on_remote_asg_client(os.path.join(save_dir, f"path_planner_cropped.mp4"), os.path.join(self.save_dir, f"path_planner_cropped.mp4"))

            # self.save_file_on_remote_asg_client(os.path.join(save_dir, f"path_planner_vehicle_centered.avi"), os.path.join(self.save_dir, f"path_planner_vehicle_centered.avi"))
            # self.save_file_on_remote_asg_client(os.path.join(save_dir, f"path_planner_vehicle_centered.mp4"), os.path.join(self.save_dir, f"path_planner_vehicle_centered.mp4"))
    
        self.save_pickle_object_on_asg_client(self.map_snapshots, "astar_map_snapshots.pkl")

    def save_node_data(self):
        if not self.b_save_minimal:
            self.save_trav_semseg_video()
            self.save_path_planner_video()

        vehicle_enabled_duration = stamp_utils.get_stamp_dt(self.vehicle_enabled_timestamp, self.vehicle_disabled_timestamp)
        active_camera_processing_duration = stamp_utils.get_stamp_dt(self.map_snapshots[0].stamp, self.vehicle_disabled_timestamp)
        num_expected_images = round(self.expected_camera_frame_rate * active_camera_processing_duration)
        
        # Save snapshots summary
        self.snapshots_summary.vehicle_enabled_duration = vehicle_enabled_duration
        self.snapshots_summary.active_camera_processing_duration = active_camera_processing_duration
        self.snapshots_summary.expected_camera_frame_rate = self.expected_camera_frame_rate
        self.snapshots_summary.num_expected_camera_images = num_expected_images
        self.snapshots_summary.num_images_processed = len(self.snapshots_summary.image_processing_duration)

        # Populate data from 2D planner summary
        planner_snapshots_summary : AstarSnapshotsSummary = self.load_pickle_object_from_asg_client("astar_2D_planner_snapshots_summary.pkl")
        self.snapshots_summary.active_path_replanning_duration = planner_snapshots_summary.active_path_replanning_duration
        self.snapshots_summary.astar_iterations_per_replanning_step = planner_snapshots_summary.astar_iterations_per_replanning_step
        self.snapshots_summary.astar_replanning_step_duration = planner_snapshots_summary.astar_replanning_step_duration
        self.snapshots_summary.expected_path_replanning_rate = planner_snapshots_summary.expected_path_replanning_rate
        self.snapshots_summary.num_expected_replanning_steps = planner_snapshots_summary.num_expected_replanning_steps
        self.snapshots_summary.num_replanning_steps = planner_snapshots_summary.num_replanning_steps
        self.save_pickle_object_on_asg_client(self.snapshots_summary, "astar_snapshots_summary.pkl")

        self.log("info", f"NUM COLOR IMAGES = {self.num_color}")
        self.log("info", f"NUM DEPTH IMAGES = {self.num_depth}")
        camera_dt = stamp_utils.get_stamp_dt(self.map_snapshots[0].stamp, self.map_snapshots[-1].stamp)
        self.log("info", f"PROCESSED {len(self.map_snapshots)} snapshots in {self.image_proc_time:.2f} sec (avg rate {self.image_proc_time/len(self.map_snapshots):.4f} sec/snapshot) over {camera_dt:.2f} seconds of sensor data")
        # self.log("info", f"First few map state processing times: {self.snapshots_summary.image_processing_duration[:9]} seconds.")
        max_proc_sim_time = stamp_utils.get_stamp_dt(self.initial_cam_proc_stamp, self.max_proc_stamp)
        self.log("info", f"Max img processing duration {self.max_proc_time:.4f} sec occurred at t = {max_proc_sim_time:.2f} sec")
        self.log("info", f"EXPECTED {num_expected_images} images over {active_camera_processing_duration:.2f} seconds")

        self.log("info", "Saved node data")

    def check_for_rerun_request(self):        
        # Failed to process appropriate amount of camera images
        if math.fabs(self.snapshots_summary.num_images_processed - self.snapshots_summary.num_expected_camera_images) > round(0.05 * self.snapshots_summary.num_expected_camera_images):
            self.notify_ready_request.request_rerun = True
            self.notify_ready_request.reason_for_rerun = f"Processed too few camera images. Processed {self.snapshots_summary.num_images_processed} but expected {self.snapshots_summary.num_expected_camera_images} over {self.snapshots_summary.active_camera_processing_duration:.2f} seconds."
            self.log("warn", f"Requesting rerun with reason: {self.notify_ready_request.reason_for_rerun}")


def main(args=None):
    # Set memory growth to True
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    # Place the main directory inside the src folder
    node_name = 'astar_trav_vehicle'
    path_prefix = os.getcwd()
    if "/src" not in path_prefix:
        path_prefix = os.path.join(path_prefix, "src")

    semantic_color_map = ((255, 255, 255), # 0 Traversable
                            (0, 0, 0),      # 1 Non-traversable
                            (0, 0, 255))    # 2 Sky

    # Dataset_Barberry_Bushes: Barberry bushes, all nontraversable
    # Dataset_Trees_Bushes: Juniper trees and barberry bushes, all nontraversable.
    #   - Sunlight inclination was about 31.34 deg.
    # Dataset_Trees_Notrav_Bushes_Trav: Juniper trees nontraversable and barberry bushes traversable

    semseg_dict = {
        "mode": "prediction",
        "data_dir": os.path.join(path_prefix, "astar_trav", "astar_trav", "Datasets_256x256", "NoClouds_Trees_Bushes_SunIncl0-15-30-45-60-75-90"),
        "model_version": "small_v2",
        "image_size": (256, 256, 3),
        "semantic_color_map": semantic_color_map,
        "create_network_func": semseg_networks.create_unet_network_small_v2,
        "seg_folder_name": "trav"
    }

    path_planner_dict = {
        "env_size": (60., 60.),
        "border_size": 50.,
        "minor_cell_size": 1.,
        "major_cell_size": 2.,
        "bell_curve_sigma": 1.5, # 2.5 in v1, 1.5 in v2-v4
        "bell_curve_alpha": 1/50.,
        "max_votes_per_vertex": 50,
        "radius_threashold1": 3.,
        "radius_threashold2": 6.,
        "use_three_point_edge_cost": False,
    }

    # Sensor offsets and orientations w.r.t vehicle center
    cam_offset = np.array([1.5, 0., 1.4], dtype=np.float32)                           # Offset is (x, y, z) in [m]
    cam_orientation = np.array([0., 0., 0.], dtype=np.float32) * (math.pi / 180.)   # Orientation is (yaw, pitch, roll) in [rad] based on the ZYX sequence
    loc_offset = np.array([0., 0., 0.], dtype=np.float32)                           # Offset is (x, y, z) in [m]
    loc_orientation = np.array([0., 0., 0.], dtype=np.float32) * (math.pi / 180.)   # Orientation is (yaw, pitch, roll) in [rad] based on the ZYX sequence

    astar_trav_mapping_dict = {
        "height_thresh": 2.,                                                            # Height threshold for depth data [m]
        "camera_hfov": 90. * (math.pi / 180.),                                          # Camera horizontal FOV [rad]
        "veh2cam_transform": {"offset": cam_offset, "orientation": cam_orientation},    # Vehicle center to camera transform
        "veh2loc_transform": {"offset": loc_offset, "orientation": loc_orientation},    # Vehicle center to localization sensor transform
        "initial_sensor_proc_delay": 1.,
        "b_use_vectorized_obstacle_vote_update": True,
        "nav_destination_topic_suffix": "/nav/destination",
        "localization_topic_suffix": "/sensors/localization",
        "color_cam_topic_suffix": "/sensors/camera/color_image",
        "depth_cam_topic_suffix": "/sensors/camera/depth_image",
        "map_state_topic_suffix": "/nav/map_state",
        "expected_camera_frame_rate": 15.,
    }

    rclpy.init(args=args)
    node = AstarTrav2DMapper(node_name, semseg_dict, path_planner_dict, **astar_trav_mapping_dict)
    spin_vehicle_node(node, num_threads=2)