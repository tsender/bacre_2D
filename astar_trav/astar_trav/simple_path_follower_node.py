import math
import os
import pickle
import time
import numpy as np
import transforms3d
import traceback
from typing import List, Tuple

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import rclpy
import rclpy.node
import rclpy.executors
from rclpy.executors import _rclpy
import rclpy.handle
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs

import auto_scene_gen_msgs.msg as auto_scene_gen_msgs
from auto_scene_gen_core.vehicle_node import AutoSceneGenVehicleNode, VehicleNodeException, spin_vehicle_node
import auto_scene_gen_core.stamp_utils as stamp_utils

def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))

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

class PathFollowerTrackingSnapshot:
    def __init__(self):
        self.path_waypoints = None
        self.tracking_waypoint = None
        self.vehicle_position = None
        self.vehicle_orientation = None
        self.control_velocity = 0.          # [m/s]
        self.control_steering_angle = 0.    # [rad]
        self.stamp = None


class PathFollowerSnapshotsSummary:
    def __init__(self):
        self.vehicle_enabled_duration = None
        self.active_control_duration = None
        self.expected_control_rate = None
        self.actual_control_rate = None
        self.control_processing_duration : List[float] = []
        self.num_control_messages = None
        self.num_waypoint_messages = None


class SimplePathFollower(AutoSceneGenVehicleNode):
    def __init__(self, 
                node_name: str,
                tracking_speed: float,
                kp_steering: float,
                kd_steering: float,
                control_timer_rate: float,
                tracking_waypoint_method: str,
                waypoint_mixture_coeffs: List[int],
                look_ahead_distance: float,
                localization_topic_suffix: str,
                waypoint_path_topic_suffix: str,
                vehicle_control_topic_suffix: str,
                ):
        super().__init__(node_name)

        # Keep track of current and tracking waypoints
        self.tracking_waypoints = [] # List of 2D numpy arrays
        self.vehicle_path = [] # List of 2D numpy arrays
    
        # Main params
        self.tracking_speed = tracking_speed # [m/s]
        self.control_timer_rate = control_timer_rate
        self.control_timer_dt = 1./control_timer_rate
        self.vehicle_position = None
        self.vehicle_orientation = None
        self.current_yaw = 0.
        self.waypoints = None
        self.vehicle_position_stamp = None

        if tracking_waypoint_method not in ['waypoint_mixture_coeffs', 'look_ahead_distance']:
            raise ValueError(f"Tracking waypoint method {tracking_waypoint_method} not recognized.")
        self.tracking_waypoint_method = tracking_waypoint_method
        
        if len(waypoint_mixture_coeffs) != 3 or math.fabs(sum(waypoint_mixture_coeffs) - 1) > 0.01:
            raise ValueError(f"Waypoint mixture coeffs must contain three elements that sum to 1.")
        self.waypoint_mixture_coeffs = waypoint_mixture_coeffs
        self.look_ahead_distance = look_ahead_distance

        self.kp_steering = kp_steering
        self.kd_steering = kd_steering
        self.desired_yaw = 0.
        self.last_error = None
        self.tracking_snapshots : List[PathFollowerTrackingSnapshot] = []
        self.snapshots_summary = PathFollowerSnapshotsSummary()

        sensor_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Subs
        self.loc_topic = self.vehicle_topic_prefix + localization_topic_suffix
        self.waypoint_topic = self.vehicle_topic_prefix + waypoint_path_topic_suffix
        self.localization_sub = self.create_subscription(geometry_msgs.PoseStamped, self.loc_topic, self.localization_cb, 10)
        self.waypoint_path_sub = self.create_subscription(nav_msgs.Path, self.waypoint_topic, self.waypoint_path_cb, 10)

        # Pubs
        self.vehicle_control_topic = self.vehicle_topic_prefix + vehicle_control_topic_suffix
        self.vehicle_control_pub = self.create_publisher(auto_scene_gen_msgs.PhysXControl, self.vehicle_control_topic, 10)

        # Timers
        self.control_timer = None # Gets created when we receive the first waypoints message

        self.log("info", f'Initialized simple path follower for worker ID {self.wid}.')

        self.last_waypoint_stamp = None
        self.num_waypoint_msgs_received = 0
        self.num_control_messages_sent = 0
        self.first_control_stamp = None

    def reset(self):
        self.vehicle_position = None
        self.vehicle_orientation = None
        self.waypoints = None
        self.last_error = None
        self.tracking_snapshots.clear()
        self.snapshots_summary = PathFollowerSnapshotsSummary()

        for i in range(3):
            if self.destroy_timer(self.control_timer):
                self.log("info", "Destroyed control timer")
                break
            else:
                self.log("warn", f"Could not destroy control timer on attempt {i+1}")
                time.sleep(0.1)
        time.sleep(0.5)
        self.control_timer = None

        self.last_waypoint_stamp = None
        self.num_waypoint_msgs_received = 0
        self.num_control_messages_sent = 0
        self.first_control_stamp = None
        self.log("info", f"Path follower has been reset.")
    
    def localization_cb(self, msg: geometry_msgs.PoseStamped):
        if not self.vehicle_ok():
            return

        self.vehicle_position = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)
        self.vehicle_orientation = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z] # Let's assume aligned with vehicle frame
        euler = transforms3d.euler.quat2euler(self.vehicle_orientation, 'rzyx')
        self.current_yaw = euler[0]
        self.vehicle_position_stamp = msg.header.stamp

    def waypoint_path_cb(self, msg: nav_msgs.Path):
        """Update list of waypoints with new list of waypoints"""
        if not self.vehicle_ok():
            return

        if self.last_waypoint_stamp is None:
            self.num_waypoint_msgs_received += 1
        elif not stamp_utils.do_ros_stamps_match(self.last_waypoint_stamp, msg.header.stamp):
            self.num_waypoint_msgs_received += 1
        self.last_waypoint_stamp = msg.header.stamp

        self.waypoints = []
        for pose_msg in msg.poses:
            p = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y], dtype=np.float32)
            self.waypoints.append(p)

        # Createe/reset timer
        if self.control_timer is None:
            self.log("info", "Creating control timer")
            self.control_timer = self.create_timer(self.control_timer_dt, self.control_timer_cb, clock=self.sim_clock) # This will begin the timer
            self.control_timer_cb() # Call now since creating the new timer won't call it yet

    def compute_tracking_waypoint(self):
        num_waypoints = len(self.waypoints)
        closest_idx = None
        min_dist = math.inf
        for i in range(num_waypoints):
            dist = fast_norm(self.vehicle_position - self.waypoints[i])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Compute next waypoint for tracking
        if closest_idx == num_waypoints - 1: # Closest point is the goal point
            return self.waypoints[-1]
        else:
            if self.tracking_waypoint_method == 'waypoint_mixture_coeffs':
                tracking_point = self.waypoint_mixture_coeffs[0] * self.vehicle_position + \
                                 self.waypoint_mixture_coeffs[1] * self.waypoints[closest_idx] + \
                                 self.waypoint_mixture_coeffs[2] * self.waypoints[closest_idx + 1]
                return tracking_point

            elif self.tracking_waypoint_method == 'look_ahead_distance':
                idx1 = None
                for i in range(closest_idx, num_waypoints-1):
                    # Find waypoint[i] that is within look-ahead distance while waypoint[i+1] is farther than look-ahead distance away from vehicle
                    if fast_norm(self.waypoints[i] - self.vehicle_position) < self.look_ahead_distance and fast_norm(self.waypoints[i+1] - self.vehicle_position) > self.look_ahead_distance:
                        idx1 = i
                        break
                if idx1 is None:
                    idx1 = closest_idx

                # If both segment endpoints are inside look-ahead radius, keep incrementing idx1 until the segment interesects the radius
                while (idx1 < num_waypoints - 1) and fast_norm(self.waypoints[idx1] - self.vehicle_position) < self.look_ahead_distance and fast_norm(self.waypoints[idx1+1] - self.vehicle_position) < self.look_ahead_distance:
                    idx1 += 1
                
                if idx1 == num_waypoints - 1:
                    return self.waypoints[-1]
                else:
                    return find_look_ahead_point_on_segment(self.vehicle_position, self.waypoints[idx1], self.waypoints[idx1+1], self.look_ahead_distance)

    def control_timer_cb(self):
        """Find the closest waypoint to the current location and send a control command for the next waypoint"""
        if not self.vehicle_ok() or self.vehicle_position is None or self.vehicle_orientation is None or self.waypoints is None:
            return

        start_time = time.time()
        if self.first_control_stamp is None:
            self.first_control_stamp = self.sim_clock.now().to_msg()

        tracking_waypoint = self.compute_tracking_waypoint()

        delta_pos = tracking_waypoint - self.vehicle_position
        self.desired_yaw = math.atan2(delta_pos[1], delta_pos[0])
        error = self.desired_yaw - self.current_yaw
        if error > math.pi:
            error -= 2*math.pi
        elif error < -math.pi:
            error += 2*math.pi

        dedt = 0.
        if self.last_error is not None:
            dedt = (error - self.last_error) / self.control_timer_dt
        self.last_error = error
        steering_angle = self.kp_steering * error + self.kd_steering * dedt # [deg]

        control_proc_duration = time.time() - start_time

        stamp = self.sim_clock.now().to_msg()
        msg = auto_scene_gen_msgs.PhysXControl()
        msg.header.stamp = stamp
        msg.header.frame_id = "/vehicle"
        msg.longitudinal_velocity = self.tracking_speed * 100. # [cm/s]
        msg.steering_angle = -steering_angle # [deg], Need to flip the sign b/c positive steering for PhysX is to the right
        msg.handbrake = False
        self.vehicle_control_pub.publish(msg)

        self.num_control_messages_sent += 1
        self.snapshots_summary.control_processing_duration.append(control_proc_duration)
        
        # Add snapshot
        snapshot = PathFollowerTrackingSnapshot()
        snapshot.path_waypoints = self.waypoints
        snapshot.tracking_waypoint = tracking_waypoint
        snapshot.vehicle_position = self.vehicle_position
        snapshot.vehicle_orientation = self.vehicle_orientation
        snapshot.control_velocity = self.tracking_speed # [m/s]
        snapshot.control_steering_angle = steering_angle * (math.pi / 180.) # [rad]
        snapshot.stamp = stamp
        self.tracking_snapshots.append(snapshot)

    def save_node_data(self):
        self.save_pickle_object_on_asg_client(self.tracking_snapshots, "path_follower_snapshots.pkl")

        self.snapshots_summary.vehicle_enabled_duration = stamp_utils.get_stamp_dt(self.vehicle_enabled_timestamp, self.vehicle_disabled_timestamp)
        self.snapshots_summary.active_control_duration = stamp_utils.get_stamp_dt(self.first_control_stamp, self.vehicle_disabled_timestamp)
        self.snapshots_summary.expected_control_rate = self.control_timer_rate
        self.snapshots_summary.actual_control_rate = (self.num_control_messages_sent - 1) / self.snapshots_summary.active_control_duration
        self.snapshots_summary.num_control_messages = len(self.snapshots_summary.control_processing_duration)
        self.snapshots_summary.num_waypoint_messages = self.num_waypoint_msgs_received
        self.save_pickle_object_on_asg_client(self.snapshots_summary, "path_follower_snapshots_summary.pkl")

        self.log("info", f"RECEIVED {self.num_waypoint_msgs_received} waypoint messages")
        self.log("info", f"SENT {self.num_control_messages_sent} control messages over {self.snapshots_summary.active_control_duration:.2f} seconds (avg. rate {self.snapshots_summary.actual_control_rate:.2f} cmd/sec)")
        self.log("info", f"First tracking point {self.tracking_snapshots[0].tracking_waypoint}")
        self.log("info", "Saved node data")

    def check_for_rerun_request(self):
        if math.fabs(self.snapshots_summary.actual_control_rate - self.control_timer_rate) > 0.05*self.control_timer_rate:
            self.notify_ready_request.request_rerun = True
            self.notify_ready_request.reason_for_rerun = f"Control rate was off by more than 5%. Actual rate was {self.snapshots_summary.actual_control_rate:.2f} Hz but expected {self.control_timer_rate:.2f} Hz."
            self.log("warn", f"Requesting rerun with reason: {self.notify_ready_request.reason_for_rerun}")


def main(args=None):
    node_name = 'simple_path_follower'

    # TODO: Add veh2loc offset (for now it is 0)

    pf_dict = {
        'tracking_speed': 5., # [m/s]
        'kp_steering': 10., # 10 in v1, 20 in v3
        'kd_steering': 5,
        'control_timer_rate': 20.,
        'tracking_waypoint_method': 'look_ahead_distance',   # Options: waypoint_mixture_coeffs, look_ahead_distance
        'waypoint_mixture_coeffs': [0.25, 0., 0.75], # Coeffs for [vehicle_position, closest_waypoint, next_closest_waypoint]
        'look_ahead_distance': 5.,
        'localization_topic_suffix': '/sensors/localization',
        'waypoint_path_topic_suffix': '/nav/waypoint_path',
        'vehicle_control_topic_suffix': '/control/physx'
    }

    rclpy.init(args=args)
    path_follower = SimplePathFollower(node_name, **pf_dict)
    spin_vehicle_node(path_follower, num_threads=2)