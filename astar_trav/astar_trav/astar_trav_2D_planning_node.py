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

import astar_trav.semseg_networks as semseg_networks
from astar_trav.astar_voting_path_planner import AstarVotingPathPlanner
from astar_trav.astar_trav_2D_all_in_one_node import AstarPathSnapshot, AstarSnapshotsSummary

def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))

def linear_interp(x1, x2, r):
    return x1 + r * (x2 - x1)

def find_look_ahead_point_on_segment(p: np.float32, w1: np.float32, w2: np.float32, distance: float, tol: float = 1e-5):
    """Find the point on the segment w1-w2 that is a certain distance from p.
    If two such points exist, this method will return the one closest to w2.
    If no such point exists, then it will return the point on the segment w1-w2 closest to p.
    
    Args:
        - p: Point to look ahead from, 1-D numpy array in the form of (x,y) [m]
        - w1: First segment endpoint, 1-D numpy in the form of (x,y) [m]
        - w2: Second segment endpoint, 1-D numpy in the form of (x,y) [m]
        - distance: The distance [m] the look ahead point should be from point p
        - tol: If the points are effectively colinear with this angle tolerance, then just use interpolation

    Returns:
        - The appropriate look-ahead point.
    """
    seg_w1w2 = w2 - w1 # Segment w1-w2
    seg_w1p = p - w1 # Segment w1-p
    unit_w1w2 = seg_w1w2 / fast_norm(seg_w1w2)
    unit_w1p = seg_w1p / fast_norm(seg_w1p)
    theta1 = np.arccos(np.clip(unit_w1w2.dot(unit_w1p), -1., 1.)) # Angle p-w1-w2

    # If the points are effectively colinear, just do basic interpolation along p-w2
    if math.fabs(theta1 - math.pi) <= tol or math.fabs(theta1) <= tol:
        # if fast_norm(p - w2) >= distance:
        #     seg_pw2 = w2-p # Segment p-w2
        #     unit_pw2 = seg_pw2 / fast_norm(seg_pw2)
        #     return p + unit_pw2 * distance
        # else:
        #     return w2
        d12 = fast_norm(w1-w2)
        if (fast_norm(p-w1) <= d12 and fast_norm(p-w2) <= d12) or fast_norm(p-w1) <= distance:
            seg_pw2 = w2-p # Segment p-w2
            unit_pw2 = seg_pw2 / fast_norm(seg_pw2)
            return p + unit_pw2 * distance
        elif fast_norm(p-w1) < fast_norm(p-w2):
            return w1
        else:
            return w2

    temp = (fast_norm(seg_w1p) / distance) * np.sin(theta1)
    if math.fabs(temp) <= 1.: # Okay to proceed, as at least one point exists
        # If two triangles exist, this method will always return the point closer to w2
        theta2 = np.arcsin(temp) # Angle w1-x-p, whhere x lies on segment w1-w2
        theta3 = math.pi - theta1 - theta2 # Angle w1-p-x
        dist_w1x = distance * np.sin(theta3) / np.sin(theta1)
        return w1 + unit_w1w2 * dist_w1x
    else: # No point exists, return point on w1-w2 closest to p
        if theta1 <= math.pi/2.: # Acute or right angle
            dist_w1x = fast_norm(seg_w1p) * np.cos(theta1)
            return w1 + unit_w1w2 * dist_w1x
        else:
            return w1

def get_closest_waypoint_idx(vehicle_position, waypoints):
    num_waypoints = len(waypoints)
    closest_idx = None
    min_dist = math.inf
    for i in range(num_waypoints):
        dist = fast_norm(vehicle_position - waypoints[i])
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx    

def compute_tracking_waypoint(vehicle_position, waypoints, look_ahead_distance):
        num_waypoints = len(waypoints)
        closest_idx = None
        min_dist = math.inf
        for i in range(num_waypoints):
            dist = fast_norm(vehicle_position - waypoints[i])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Compute next waypoint for tracking
        if closest_idx == num_waypoints - 1: # Closest point is the goal point
            return waypoints[-1]
        else:
            idx1 = None
            for i in range(closest_idx, num_waypoints-1):
                # Find waypoint[i] that is within look-ahead distance while waypoint[i+1] is farther than look-ahead distance away from vehicle
                if fast_norm(waypoints[i] - vehicle_position) < look_ahead_distance and fast_norm(waypoints[i+1] - vehicle_position) > look_ahead_distance:
                    idx1 = i
                    break
            if idx1 is None:
                idx1 = closest_idx

            # If both segment endpoints are inside look-ahead radius, keep incrementing idx1 until the segment interesects the radius
            while (idx1 < num_waypoints - 1) and fast_norm(waypoints[idx1] - vehicle_position) < look_ahead_distance and fast_norm(waypoints[idx1+1] - vehicle_position) < look_ahead_distance:
                idx1 += 1
            
            if idx1 == num_waypoints - 1:
                return waypoints[-1]
            else:
                return find_look_ahead_point_on_segment(vehicle_position, waypoints[idx1], waypoints[idx1+1], look_ahead_distance)

class AstarTrav2DPlanner(AutoSceneGenVehicleNode):
    PLOT_PADDING = 5.
    
    def __init__(self, 
                node_name: str, 
                path_planner_dict: Dict,
                waypoint_separation: float,                             # Used to distribute waypoints into a finer path                    
                replan_path_timer_rate: float,                           # Frequency at which to call the replan_path_cb() function
                b_use_multithreaded_path_update: bool,
                b_use_yaw_radius_threasholds: False,                    # Indictae if we should use the radius thresholds in the A* path planner
                b_generate_path_from_look_ahead_prediction: False,
                look_ahead_distance: float,
                nav_destination_topic_suffix: str,
                localization_topic_suffix: str,
                map_state_topic_suffix: str,                             # Expected camera frame rtae in [Hz]
                waypoint_path_topic_suffix: str,
                ):
        super().__init__(node_name)

        self.path_planner = AstarVotingPathPlanner(**path_planner_dict)
        self.landscape_size = path_planner_dict['env_size']
        self.waypoint_separation = waypoint_separation

        # Current sensor data
        self.vehicle_position = None # Vehicle position
        self.vehicle_orientation = None # Vehicle orientation

        self.goal_point = None
        self.initial_map_state_stamp = None
        self.initial_astar_replan_stamp = None
        self.b_use_multithreaded_path_update = b_use_multithreaded_path_update
        self.b_use_yaw_radius_threasholds = b_use_yaw_radius_threasholds
        self.b_generate_path_from_look_ahead_prediction = b_generate_path_from_look_ahead_prediction
        self.look_ahead_distance = look_ahead_distance

        self.max_votes_per_vertex = path_planner_dict['max_votes_per_vertex']
        self.latest_obstacle_votes = None
        self.latest_obstacle_votes_compact_np = None
        self.latest_obstacle_locations_compact_np = None
        self.last_path = None

        # Variables for saving data
        self.path_snapshots : List[AstarPathSnapshot] = []
        self.snapshots_summary = AstarSnapshotsSummary()

        sensor_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Subs
        self.nav_topic = self.worker_topic_prefix + nav_destination_topic_suffix
        self.loc_topic = self.vehicle_topic_prefix + localization_topic_suffix
        self.map_state_topic = self.vehicle_topic_prefix + map_state_topic_suffix
        self.nav_destination_sub = self.create_subscription(geometry_msgs.Pose, self.nav_topic, self.nav_destination_cb, 10)
        self.localization_sub = self.create_subscription(geometry_msgs.PoseStamped, self.loc_topic, self.localization_cb, 10)
        self.map_state_sub = self.create_subscription(astar_trav_msgs.MapState2D, self.map_state_topic, self.map_state_cb, 10)

        # Pubs
        self.waypoint_topic = self.vehicle_topic_prefix + waypoint_path_topic_suffix
        self.waypoint_path_pub = self.create_publisher(nav_msgs.Path, self.waypoint_topic, 10)

        # Timers
        self.replan_path_timer_cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup() # Must be mutually exclusive to avoid race conditions
        self.replan_path_timer_rate = replan_path_timer_rate
        self.replan_path_timer = None # Create timer when ready

        # Threading stuff
        self.lock = threading.Lock()
        self.replan_event = threading.Event()

        self.log("info", f'Initialized A* trav planning node for worker ID {self.wid}.')

        # Debug variables, used in logging
        self.num_map_states_received = 0
        self.astar_proc_time = 0.
        self.num_waypoint_msgs_sent = 0
    
    def reset(self):
        self.log("info", f"Waiting for replan event to be set...")
        self.replan_event.wait()

        self.path_planner.clear_map()
        self.goal_point = None
        self.last_path = None

        self.vehicle_position = None
        self.vehicle_orientation = None
        self.initial_map_state_stamp = None
        self.initial_astar_replan_stamp = None

        self.latest_obstacle_votes = None
        self.latest_obstacle_votes_compact_np = None
        self.latest_obstacle_locations_compact_np = None

        for i in range(3):
            if self.destroy_timer(self.replan_path_timer):
                self.log("info", "Destroyed replan path timer.")
                break
            else:
                self.log("warn", f"Could not destroy replan path timer on attempt {i+1}")
                time.sleep(0.1)
        time.sleep(0.5)
        self.replan_path_timer = None

        self.path_snapshots.clear()
        self.snapshots_summary = AstarSnapshotsSummary()

        self.num_map_states_received = 0
        self.astar_proc_time = 0.
        self.num_waypoint_msgs_sent = 0
        self.log("info", f"2D planner has been reset.")

    def nav_destination_cb(self, msg: geometry_msgs.Pose):
        if not self.vehicle_ok():
            return
        
        with self.lock:
            self.goal_point = self.path_planner.get_nearest_vertex(np.array([msg.position.x, msg.position.y]), b_major=True)
    
    def localization_cb(self, msg: geometry_msgs.PoseStamped):
        if not self.vehicle_ok():
            return
        
        # TODO: Account for veh2loc_offset from localization sensor (for now it is 0)
        with self.lock:
            self.vehicle_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32).reshape(3,1) # - self.veh2loc_offset
            self.vehicle_orientation = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] # Let's assume aligned with vehicle frame
            self.R_veh2w = transforms3d.quaternions.quat2mat(self.vehicle_orientation) # Rotation matrix body to world
            self.euler = transforms3d.euler.quat2euler(self.vehicle_orientation, 'rzyx')

    def map_state_cb(self, msg: astar_trav_msgs.MapState2D):
        if not self.vehicle_ok():
            return
        
        if msg.env_size_x != self.path_planner.env_size[0] or msg.env_size_y != self.path_planner.env_size[1] or msg.minor_cell_size != self.path_planner.minor_cell_size[0]:
            self.log("warn", f"Incoming MapState2D message does not match expected parameters. env_size {(msg.env_size_x, msg.env_size_y)} != {self.path_planner.env_size}, and minor cell size {msg.minor_cell_size} != {self.path_planner.minor_cell_size[0]}.")
            return
        
        if len(msg.obstacle_votes) > 0 and len(msg.obstacle_votes) != self.path_planner.obstacle_votes.shape[0]:
            self.log("warn", f"Incoming MapState2D message field 'obstacle_votes' has length {len(msg.obstacle_votes)} but expected {self.path_planner.obstacle_votes.shape[0]}.")
            return
        
        if self.initial_map_state_stamp is None:
            self.initial_map_state_stamp = self.sim_clock.now().to_msg()

        # Store latest obstacle info
        with self.lock:
            if len(msg.obstacle_votes):
                self.latest_obstacle_votes = np.array(msg.obstacle_votes)
                self.latest_obstacle_votes_compact_np = np.array(msg.obstacle_votes_compact)
                self.latest_obstacle_locations_compact_np = np.array([msg.obstacle_locations_compact_x, msg.obstacle_locations_compact_y]).T

        self.num_map_states_received += 1

        if self.replan_path_timer is None:
            self.initial_astar_replan_stamp = self.sim_clock.now().to_msg()

            # Create/reset timer
            self.log("info", "Creating replan path timer")
            if self.b_use_multithreaded_path_update:
                self.replan_path_timer = self.create_timer(1./self.replan_path_timer_rate, self.replan_path, callback_group=self.replan_path_timer_cb_group, clock=self.sim_clock)
            else:
                self.replan_path_timer = self.create_timer(1./self.replan_path_timer_rate, self.replan_path, clock=self.sim_clock)
            self.replan_path() # Call now since creating the new timer won't call it yet

    def replan_path(self):
        """This is the main piece of code for the A* traversability path planner"""
        if not self.vehicle_ok() or self.goal_point is None or self.vehicle_position is None or self.num_map_states_received == 0:
            return

        # Get a copy of necessary data
        with self.lock:
            path_stamp = self.sim_clock.now().to_msg()
            vehicle_position = self.vehicle_position
            vehicle_orientation = self.vehicle_orientation
            euler = self.euler
            goal_point = self.goal_point

            # Override path planner obstacle info
            if self.latest_obstacle_votes is None:
                self.path_planner.update_compact_numpy_arrays()
            else:
                self.path_planner.obstacle_votes = self.latest_obstacle_votes.copy()
                self.path_planner.obs_votes_compact_np = self.latest_obstacle_votes_compact_np.copy()
                self.path_planner.obs_locations_compact_np = self.latest_obstacle_locations_compact_np.copy()

        if not self.vehicle_ok() or goal_point is None or vehicle_position is None:
            return

        self.log("info", f"Running A*...")
        self.replan_event.clear() # Reset event here
        start_time = time.time()
        start_vertex = self.path_planner.get_nearest_vertex(vehicle_position.flatten()[0:2], b_major=True)
        if self.b_generate_path_from_look_ahead_prediction and self.last_path is not None:
            look_ahead_point = compute_tracking_waypoint(vehicle_position.flatten()[0:2], self.last_path, self.look_ahead_distance)
            closest_look_ahead_idx = get_closest_waypoint_idx(look_ahead_point, self.last_path)

            if closest_look_ahead_idx == len(self.last_path) - 1:
                if self.b_use_yaw_radius_threasholds:
                    opt_path, opt_cost, iterations = self.path_planner.construct_optimal_path(start_vertex, goal_point, start_yaw=euler[0], event=self.vehicle_disabled_event)
                else:
                    opt_path, opt_cost, iterations = self.path_planner.construct_optimal_path(start_vertex, goal_point, start_yaw=None, event=self.vehicle_disabled_event)
            else:
                # Save initial portion of last trajectory
                closest_idx = get_closest_waypoint_idx(vehicle_position.flatten()[0:2], self.last_path)
                opt_path = []
                idx = closest_idx
                while idx != closest_look_ahead_idx:
                    opt_path.append(self.last_path[idx])
                    idx += 1

                # Create new trajectory beginning at look ahead point
                dv = self.last_path[closest_look_ahead_idx+1] - self.last_path[closest_look_ahead_idx]
                look_ahead_yaw = math.atan2(dv[1], dv[0])

                if self.b_use_yaw_radius_threasholds:
                    opt_path2, opt_cost2, iterations = self.path_planner.construct_optimal_path(self.last_path[closest_look_ahead_idx], goal_point, start_yaw=look_ahead_yaw, event=self.vehicle_disabled_event)
                else:
                    opt_path2, opt_cost2, iterations = self.path_planner.construct_optimal_path(self.last_path[closest_look_ahead_idx], goal_point, start_yaw=None, event=self.vehicle_disabled_event)
                
                # If the vehicle disabled event is set, then opt_path2 will be None
                if opt_path2 is not None:
                    opt_path += opt_path2
                opt_cost = self.path_planner.get_arbitrary_path_cost(opt_path)
        else:
            if self.b_use_yaw_radius_threasholds:
                opt_path, opt_cost, iterations = self.path_planner.construct_optimal_path(start_vertex, goal_point, start_yaw=euler[0], event=self.vehicle_disabled_event)
            else:
                opt_path, opt_cost, iterations = self.path_planner.construct_optimal_path(start_vertex, goal_point, start_yaw=None, event=self.vehicle_disabled_event)
        self.last_path = opt_path
        self.replan_event.set() # Set event so the main thread knows we finished
        
        # Don't bother sending if vehicle got disabled during A* iterations
        if not self.vehicle_ok():
            self.log("info", f"Vehicle was disabled during A* replanning. A* iterations {iterations} in {time.time() - start_time:.4f} seconds")
            self.last_path = None
            return
        
        astar_replan_time = time.time() - start_time
        self.astar_proc_time += astar_replan_time
        self.log("info", f"A* iterations {iterations} in {astar_replan_time:.4f} seconds")
        if opt_path is None:
            self.log("info", f"A* could not find path from {start_vertex} to {goal_point}, skipping iteration.")
            return
        
        # path_length = self.path_planner.compute_path_length(opt_path)
        # if self.waypoint_separation is not None and self.waypoint_separation > 0:
        #     num_waypoints = math.ceil(path_length / self.waypoint_separation) + 1
        #     distributed_waypoints = self.path_planner.distribute_path_waypoints(opt_path, num_waypoints)
        # else:
        #     distributed_waypoints = opt_path

        # Add path snapshot (used for minimal data saving)
        path_snapshot = AstarPathSnapshot()
        path_snapshot.vehicle_position = vehicle_position
        path_snapshot.vehicle_orientation = vehicle_orientation
        path_snapshot.obstacle_votes = self.path_planner.obstacle_votes
        path_snapshot.opt_path = opt_path
        path_snapshot.opt_path_cost = opt_cost
        path_snapshot.stamp = path_stamp
        path_snapshot.max_votes_per_vertex = self.path_planner.max_votes_per_vertex
        direct_path, direct_cost = self.path_planner.create_valid_path_from_waypoints([start_vertex, (start_vertex + goal_point)/2., goal_point])
        path_snapshot.direct_path = direct_path
        path_snapshot.direct_path_cost = direct_cost
        self.path_snapshots.append(path_snapshot)

        # Create list of PoseStamped msgs
        poses = []
        for i in range(len(opt_path)):
            pose_msg = geometry_msgs.PoseStamped()
            pose_msg.header.stamp = path_stamp
            pose_msg.header.frame_id = "/vehicle"
            pose_msg.pose.position.x = opt_path[i][0]
            pose_msg.pose.position.y = opt_path[i][1]
            pose_msg.pose.position.z = 0.
            pose_msg.pose.orientation.w = 1.
            pose_msg.pose.orientation.x = 0.
            pose_msg.pose.orientation.y = 0.
            pose_msg.pose.orientation.z = 0.
            poses.append(pose_msg)

        path_msg = nav_msgs.Path()
        path_msg.header.stamp = path_stamp
        path_msg.header.frame_id = "/vehicle"
        path_msg.poses = poses

        # Due to the low frequency of these path updates, send a few times to better ensure receipt
        for i in range(2):
            self.waypoint_path_pub.publish(path_msg)
        self.num_waypoint_msgs_sent += 1
        self.snapshots_summary.astar_iterations_per_replanning_step.append(iterations)
        self.snapshots_summary.astar_replanning_step_duration.append(astar_replan_time)

    def save_node_data(self):
        # Save path snapshots
        for snapshot in self.path_snapshots:
            snapshot.minor_vertices = self.path_planner.minor_vertices.copy()
        self.save_pickle_object_on_asg_client(self.path_snapshots, "astar_path_snapshots.pkl")

        # Save snapshots summary (only path planner portion)
        vehicle_enabled_duration = stamp_utils.get_stamp_dt(self.vehicle_enabled_timestamp, self.vehicle_disabled_timestamp)
        self.snapshots_summary.vehicle_enabled_duration = vehicle_enabled_duration
        self.snapshots_summary.active_path_replanning_duration = stamp_utils.get_stamp_dt(self.initial_astar_replan_stamp, self.vehicle_disabled_timestamp)
        self.snapshots_summary.expected_path_replanning_rate = self.replan_path_timer_rate
        self.snapshots_summary.num_expected_replanning_steps = math.floor(self.snapshots_summary.active_path_replanning_duration * self.snapshots_summary.expected_path_replanning_rate) + 1 # Plus 1 becasue we start at t=0
        self.snapshots_summary.num_replanning_steps = len(self.snapshots_summary.astar_iterations_per_replanning_step)
        self.save_pickle_object_on_asg_client(self.snapshots_summary, "astar_2D_planner_snapshots_summary.pkl")

        map_state_dt = stamp_utils.get_stamp_dt(self.initial_map_state_stamp, self.vehicle_disabled_timestamp)
        self.log("info", f"RECIEVED {self.num_map_states_received} map state messages over {map_state_dt:.2f} seconds")
        self.log("info", f"SENT {self.num_waypoint_msgs_sent} waypoint messages")
        self.log("info", f"Avg A* calculations {self.astar_proc_time/self.num_waypoint_msgs_sent:.4f} sec/replan step")

        self.log("info", "Saved node data")

    def check_for_rerun_request(self):        
        # Failed to run replan_path callback at correct rate
        if self.snapshots_summary.num_replanning_steps <= 1:
            raise VehicleNodeException(f"Replan timer only fired {self.snapshots_summary.num_replanning_steps} times. This is likely due to a ROS error. Killing node. Restart when ready.")
        
        if math.fabs(self.snapshots_summary.num_replanning_steps - self.snapshots_summary.num_expected_replanning_steps) > round(0.05 * self.snapshots_summary.num_expected_replanning_steps):
            self.notify_ready_request.request_rerun = True
            self.notify_ready_request.reason_for_rerun = f"Replan path timer misfired. Ran {self.snapshots_summary.num_replanning_steps} times but expected {self.snapshots_summary.num_expected_replanning_steps} times over {self.snapshots_summary.active_path_replanning_duration:.2f} seconds."
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

    astar_trav_planning_dict = {
        "waypoint_separation": -1,                                                      # (Deprecated) Used to distribute waypoints into a finer path    
        "replan_path_timer_rate": 0.5,
        "b_use_multithreaded_path_update": True,
        "b_use_yaw_radius_threasholds": True,
        "b_generate_path_from_look_ahead_prediction": True,
        "look_ahead_distance": 5.,
        "nav_destination_topic_suffix": "/nav/destination",
        "localization_topic_suffix": "/sensors/localization",
        "map_state_topic_suffix": "/nav/map_state",
        "waypoint_path_topic_suffix": "/nav/waypoint_path",
    }

    rclpy.init(args=args)
    node = AstarTrav2DPlanner(node_name, path_planner_dict, **astar_trav_planning_dict)
    spin_vehicle_node(node, num_threads=2)