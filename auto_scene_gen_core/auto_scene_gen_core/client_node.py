import math
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Tuple, List, Dict

import rclpy
import rclpy.callback_groups
import rclpy.node
from auto_scene_gen_core.scenario_builder import AutoSceneGenScenarioBuilder

import auto_scene_gen_msgs.msg as auto_scene_gen_msgs
import auto_scene_gen_msgs.srv as auto_scene_gen_srvs

# TODO: Make a cpp version of this class?

# NOTE on running ros1_bridge: 
# To list all ros1_bridge pairs: ros2 run ros1_bridge dynamic_bridge --print-pairs
# To run ros1_bridge: ros2 run ros1_bridge dynamic_bridge --bridge-all-topics

def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))

class AutoSceneGenWorkerRef:
    """This class creates an AutoSceneGenWorker reference to keep track of interactions with the actual worker in UE4.
    You may create child classes if you need to add additional functionality.
    """
    def __init__(self, wid: int):
        # Worker info
        self.wid = wid # Worker ID
        self.status = auto_scene_gen_msgs.StatusCode.OFFLINE
        self.b_registered_with_asg_client = False
        self.run_scenario_submit_time = None
        self.b_received_run_scenario_request = False
        self.b_running_scenario = False
        self.registered_vehicle_nodes = []
        self.ready_vehicle_nodes = []
        self.b_vehicle_nodes_can_run = False
        self.vehicle_node_save_dir = None
        self.b_save_minimal = True
        self.num_resubmission_attempts = 0

        self.b_vehicle_node_requested_rerun = False
        self.num_rerun_requests = 0

        # Scenario information
        self.scenario_number = 0 # This is mainly for verification purposes b/w this node and the AutoSceneGenWorker
        self.run_scenario_request = None # RunScenario request
        self.ue_run_scenario_request = None # UE RunScenario request to be sent to the AutoSceneGenWorker
        self.run_scenario_future_response = None
        self.analyze_scenario_request = None # AnalyzeScenario request from the AutoSceneGenWorker
        self.b_waiting_for_analyze_scenario_request = False
        self.b_ignore_analyze_scenario_request = False
        self.b_analyzed_scenario = False

    def set_offline(self, duration_to_subtract: float):
        """Set the worker offline
        
        Args:
            - duration_to_subtract: Duration to subtract from run_scenario_submit_time to trigger the process request timeout 
        """
        self.status = auto_scene_gen_msgs.StatusCode.OFFLINE
        self.b_registered_with_asg_client = False
        self.b_waiting_for_analyze_scenario_request = False
        self.b_ignore_analyze_scenario_request = True # Set to True just in case the client tries to make a request when the connection is reset
        self.b_received_run_scenario_request = False
        self.b_running_scenario = False

        if self.run_scenario_submit_time is None:
            self.run_scenario_submit_time = time.time() - duration_to_subtract
        else:
            self.run_scenario_submit_time -= duration_to_subtract
        
        self.num_resubmission_attempts = 0

    def reset(self):
        """Custom reset function you can override. Will be automatically called once all vehicle nodes are registered."""
        pass


class AutoSceneGenClient(rclpy.node.Node):
    """This class contains the base ROS client interface to connect to the UE4 AutomaticSceneGeneration plugin.
    """
    WORKER_PROCESS_REQUEST_TIMEOUT = 30.    # Duration before resubmitting a RunScenario request

    def __init__(self, 
                node_name: str,                                 # The name of the ROS node
                main_dir: str,                                  # The main directory for storing data on your computer
                asg_client_name: str,                           # The AutoSceneGenClient's name, which is used for creating the appropriate ROS topics (this does not need to match the ROS node name)
                num_vehicle_nodes: int,                         # Number of AutoSceneGenVehicleNodes in the AutoScenegenVehicle's the autonomy stack
                num_workers: int,                               # Number of AutoSceneGenWorkers to keep track of
                base_wid: int,                                  # The base, or starting, AutoSceneGenWorker ID
                local_wids: List[int],
                worker_class: AutoSceneGenWorkerRef,            # A class or subclass instance of AutoSceneGenWorkerRef, used for managing AutoSceneGenWorkers
                scenario_builder: AutoSceneGenScenarioBuilder,   # A class or subclass instance of AutoSceneGenScenarioBuilder, used for creating scenarios
                ):
        """All inputs should be in terms of a right-handed North-West-Up (NWU) coordinate system and in SI units, unless stated otherwise.
        However, any angular attributes used to describe the scene description should be in degrees.
        This client interface will automatically handle the conversion to the Unreal coordinate system.

        Args:
            - node_name: The name of the ROS node
            - main_dir: The main directory for storing data on your computer
            - asg_client_name: The AutoSceneGenClient's name, which is used for creating the appropriate ROS topics (this does not need to match the ROS node name)
            - num_vehicle_nodes: Number of AutoSceneGenVehicleNodes in the AutoScenegenVehicle's the autonomy stack
            - num_workers: Number of AutoSceneGenWorkers to keep track of
            - base_wid: The base, or starting, AutoSceneGenWorker ID
            - worker_class: A class or subclass instance of AutoSceneGenWorkerRef, used for managing AutoSceneGenWorkers
            - scenario_builder: A class or subclass instance of AutoSceneGenScenarioBuilder, used for creating scenarios
        """
        super().__init__(node_name)

        # Log file
        self.main_dir = main_dir
        self.log_file = os.path.join(self.main_dir, "asg_log.txt")
        self.log("info", "#" * 60, b_log_ros=False)

        # Basic AutoSceneGen params
        self.asg_client_name = asg_client_name
        self.scenario_builder = scenario_builder

        # AutoSceneGenWorkers
        self.num_vehicle_nodes = num_vehicle_nodes
        self.num_workers = num_workers
        self.base_wid = base_wid
        self.local_wids = local_wids
        self.recognized_wids = [base_wid + i for i in range(self.num_workers)] # Only recognize workers from [base_wid, ..., base_wid + num_workers - 1]
        if not issubclass(worker_class, AutoSceneGenWorkerRef):
            raise Exception(f"'worker_class' must be a subclass of AutoSceneGenWorkerRef")
        self.workers : Dict[int, AutoSceneGenWorkerRef] = {}
        for wid in self.recognized_wids:
            self.workers[wid] = worker_class(wid)
        self.worker_class = worker_class
        self.log("info", f"Configured {num_workers} AutoSceneGenWorkerRefs")

        # ROS pubs
        self.asg_client_status_pub = self.create_publisher(auto_scene_gen_msgs.StatusCode, f"/{self.asg_client_name}/status", 10)
        self.vehicle_node_operating_info_pub = self.create_publisher(auto_scene_gen_msgs.VehicleNodeOperatingInfo, f"/{self.asg_client_name}/vehicle_node_operating_info", 10)
        self.scene_description_pubs = {}
        for wid in self.recognized_wids: # Dynamically create as many pubs as needed
            self.scene_description_pubs[wid] = self.create_publisher(auto_scene_gen_msgs.SceneDescription, f"/asg_worker{wid}/scene_description", 10)

        # ROS subs
        self.worker_status_subs = {}
        for wid in self.recognized_wids: # Dynamically creatse as many worker subs as needed
            cb_func = lambda msg, self=self, wid=wid: self.process_worker_status_msg(wid, msg)
            self.worker_status_subs[wid] = self.create_subscription(auto_scene_gen_msgs.StatusCode, f"/asg_worker{wid}/status", cb_func, 10)
        
        # ROS services
        self.analyze_scenario_srv = self.create_service(auto_scene_gen_srvs.AnalyzeScenario, f"/{self.asg_client_name}/services/analyze_scenario", self.analyze_scenario_service_cb)
        self.register_vehicle_node_srv = self.create_service(auto_scene_gen_srvs.RegisterVehicleNode, f"/{self.asg_client_name}/services/register_vehicle_node", self.register_vehicle_node_service_cb)
        self.unregister_vehicle_node_srv = self.create_service(auto_scene_gen_srvs.RegisterVehicleNode, f"/{self.asg_client_name}/services/unregister_vehicle_node", self.unregister_vehicle_node_service_cb)
        self.notify_ready_srv = self.create_service(auto_scene_gen_srvs.NotifyReady, f"/{self.asg_client_name}/services/notify_ready", self.notify_ready_service_cb)
        self.worker_issue_srv = self.create_service(auto_scene_gen_srvs.WorkerIssueNotification, f"/{self.asg_client_name}/services/worker_issue_notification", self.worker_issue_service_cb)
        
        # ROS clients
        self.run_scenario_clis = {}
        for wid in self.recognized_wids: # Dynamically create as many RunScenario clients as needed
            self.run_scenario_clis[wid] = self.create_client(auto_scene_gen_srvs.RunScenario, f"/asg_worker{wid}/services/run_scenario")

        # ROS timers
        self.main_loop_cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.main_loop_timer = self.create_timer(0.01, self.main_loop_timer_cb, callback_group=self.main_loop_cb_group)
        self.main_loop_wid = self.base_wid
        self.utility_loop_timer = self.create_timer(0.01, self.utility_loop_timer_cb)

        self.b_shutting_down = False
        
        self.log("info", "Initialized AutoSceneGenClient")
        self.log("info", "-" * 60, b_log_ros=False)
    
    def log(self, log_level: str, msg: str, b_log_ros: bool = True):
        """Write information to the log file asg_log.txt and to ROS logger.
        
        Args:
            - log_level: Log level
            - msg: The string to append to the log file
            - b_log_ros: Indicates if we should log the info to ROS logger
        
        """
        # Log message to log file with date and time
        with open(self.log_file, "a") as f:
            if str.lower(log_level) == "info":
                log_file_msg = f"[{datetime.datetime.now()}] [INFO] " + msg
            elif str.lower(log_level) == "warn":
                log_file_msg = f"[{datetime.datetime.now()}] [WARN] " + msg
            elif str.lower(log_level) == "error":
                log_file_msg = f"[{datetime.datetime.now()}] [ERROR] " + msg
            else:
                log_file_msg = f"[{datetime.datetime.now()}] [INFO] " + msg
            
            f.write(log_file_msg + "\n")
        
        # Log message to ROS
        if b_log_ros:
            if str.lower(log_level) == "info":
                self.get_logger().info(f"[{datetime.datetime.now()}] " + msg)
            elif str.lower(log_level) == "warn":
                self.get_logger().warn(f"[{datetime.datetime.now()}] " + msg)
            elif str.lower(log_level) == "error":
                self.get_logger().error(f"[{datetime.datetime.now()}] " + msg)
            else:
                self.get_logger().info(f"[{datetime.datetime.now()}] " + msg)

    def publish_online_status(self):
        """Let everyone know this client is online and running. Think of this as a heartbeat."""
        msg = auto_scene_gen_msgs.StatusCode()
        msg.status = auto_scene_gen_msgs.StatusCode.ONLINE_AND_RUNNING
        self.asg_client_status_pub.publish(msg)

    def publish_vehicle_node_operating_info(self):
        """Publish high-level information that the vehicle nodes for each worker should know. 
        This mainly indicates if it's okay to run and where to save any data.
        """
        msg = auto_scene_gen_msgs.VehicleNodeOperatingInfo()
        for wid,worker in self.workers.items():
            if worker.wid != self.main_loop_wid:
                msg.worker_ids.append(worker.wid)
                msg.scenario_numbers.append(worker.scenario_number)
                msg.ok_to_run.append(worker.b_vehicle_nodes_can_run)
                save_dir = worker.vehicle_node_save_dir if worker.vehicle_node_save_dir is not None else ""
                msg.save_dirs.append(save_dir)
                msg.save_minimal.append(worker.b_save_minimal)
        self.vehicle_node_operating_info_pub.publish(msg)

    def publish_scene_descriptions(self):
        """Publish scene descriptions as separate message. This message is only intended to provide a ground truth of the scene
        so that vehicle node plots can include the true scene description.
        """
        for wid,worker in self.workers.items():
            if worker.wid != self.main_loop_wid and worker.run_scenario_request is not None:
                self.scene_description_pubs[wid].publish(worker.run_scenario_request.scene_description)

    def process_worker_status_msg(self, wid: int, msg: auto_scene_gen_msgs.StatusCode):
        """Process ASG worker status message.
        
        Args:
            - wid: Worker ID
            - msg: Status message from worker
        """
        worker = self.workers[wid]

        b_back_online = (worker.status == auto_scene_gen_msgs.StatusCode.OFFLINE and msg.status != auto_scene_gen_msgs.StatusCode.OFFLINE)
        b_still_offline = (worker.status == auto_scene_gen_msgs.StatusCode.OFFLINE and msg.status == auto_scene_gen_msgs.StatusCode.OFFLINE)
        
        if b_back_online:
            self.log("info", f"Worker {wid} is online")

        worker.status = msg.status
        now = time.time()

        if worker.status == auto_scene_gen_msgs.StatusCode.OFFLINE and not b_still_offline:
            worker.set_offline(self.WORKER_PROCESS_REQUEST_TIMEOUT)
            self.log("info", f"Worker {wid} is offline")
            return

        if worker.status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_READY or worker.status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_RUNNING:
            worker.b_registered_with_asg_client = True

        if worker.status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_RUNNING and worker.b_received_run_scenario_request:
            worker.b_running_scenario = True

    def submit_run_scenario_request(self, 
                                    wid: int, 
                                    b_resubmission_attempt: bool = False,
                                    b_reset_num_rerun_requests: bool = True, 
                                    b_convert_for_ue: bool = True):
        """Submit a RunScenario request to specified worker.
        
        Args:
            - wid: Worker ID
            - b_resubmission_attempt: Indicates if this is an attempt to resubmit the RunScenario request because the worker did not receive it.
                                      If this flag is True, then the other booleans flags are ignored. Default is False.
            - b_reset_num_rerun_requests: Indicates if we should reset the number of rerun requests. Default is True.
            - b_convert_for_ue: Indicates if we should convert the RunScenario request to the UE equivalent. Default is True.
        """
        worker = self.workers[wid]

        if not worker.b_registered_with_asg_client:
            return
        
        # If scenario number is 0, then it means the worker informed us that rosbridge was interrupted
        if b_resubmission_attempt and worker.scenario_number == 0:
            return

        if worker.run_scenario_request is not None:
            if not b_resubmission_attempt:
                worker.scenario_number += 1
            
                if b_convert_for_ue:
                    worker.ue_run_scenario_request = self.scenario_builder.get_unreal_engine_run_scenario_request(worker.run_scenario_request)
                else:
                    worker.ue_run_scenario_request = worker.run_scenario_request
                worker.ue_run_scenario_request.scenario_number = worker.scenario_number

            # Do not clear list of ready vehicle nodes if the current request is only for scene captures
            if not worker.run_scenario_request.scene_capture_only:
                worker.ready_vehicle_nodes.clear()
            
            worker.b_received_run_scenario_request = False
            worker.b_waiting_for_analyze_scenario_request = True
            worker.analyze_scenario_request = None
            worker.b_analyzed_scenario = False
            worker.run_scenario_submit_time = time.time()
            worker.run_scenario_future_response = self.run_scenario_clis[wid].call_async(worker.ue_run_scenario_request)
            worker.b_running_scenario = False
            worker.b_vehicle_node_requested_rerun = False
            worker.b_ignore_analyze_scenario_request = False # Only submitting a RunScenario request can reset this flag

            if b_resubmission_attempt:
                worker.num_resubmission_attempts += 1
                self.log("info", f"Resubmitted RunScenario request: Worker {wid} / Scenario {worker.scenario_number}")
            else:
                worker.num_resubmission_attempts = 0
                if b_reset_num_rerun_requests:
                    worker.num_rerun_requests = 0
                self.log("info", f"Submitted RunScenario request: Worker {wid} / Scenario {worker.scenario_number}")

    def check_run_scenario_response_timer_cb(self):
        """Make sure the ASG workers received the RunScenario request"""
        for wid,worker in self.workers.items():
            if worker.wid == self.main_loop_wid:
                continue
            
            if worker.run_scenario_future_response is not None:
                if worker.run_scenario_future_response.done():
                    try:
                        response = worker.run_scenario_future_response.result()
                    except Exception as e:
                        # Not sure what to do if we get an exception (although, this should almost never happen)
                        self.log("error", f"RunScenario service call to Worker {wid} failed with exception: {e}")
                        continue
                    else: # try block succeeded
                        self.log("info", f"RunScenario service call to Worker {wid} receive response: {response.received}")
                        worker.b_received_run_scenario_request = response.received
                    worker.run_scenario_future_response = None
            
            if worker.b_running_scenario: # This variable is set when processing the worker status messages
                worker.run_scenario_submit_time = None
                continue
            
            now = time.time()
            if worker.run_scenario_submit_time is not None:
                if worker.b_registered_with_asg_client and worker.status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_READY:
                    if now - worker.run_scenario_submit_time > self.WORKER_PROCESS_REQUEST_TIMEOUT:
                        if worker.scenario_number > 0 and worker.run_scenario_request is not None:
                            self.log("info", f"Worker {wid} exhausted timeout of {self.WORKER_PROCESS_REQUEST_TIMEOUT} seconds to process RunScenario request.")
                        self.submit_run_scenario_request(wid, b_resubmission_attempt=True)
                        continue

    def cancel_submitting_run_scenario_request(self, wid: int):
        """Stop trying to send the ASG worker a RunScenario request
        
        Args:
            - wid: Worke ID
        """
        worker = self.workers[wid]
        worker.run_scenario_future_response = None
        worker.run_scenario_submit_time = None

    def analyze_scenario_service_cb(self, request, response):
        """Service callback for AnalyzeScenario request from AutoSceneGenWorker. All we do is store the request so we can return a quick response."""
        wid = request.worker_id

        if wid not in self.recognized_wids:
            self.log("warn", f"Received AnalyzeScenario request from unknown worker {wid}. Ignoring request.")
            response.received = False
            return response

        worker = self.workers[wid]
        if request.scenario_number != worker.scenario_number:
            self.log("warn", f"Expected Worker {wid} AnalyzeScenario Request with scenario {worker.scenario_number} but received {request.scenario_number}.")
            self.submit_run_scenario_request(wid, b_resubmission_attempt=True)
            response.received = False
            return response
        
        if worker.b_ignore_analyze_scenario_request:
            self.log("warn", f"Worker {wid} was flagged to ignore AnalyzeScenario request. Ignoring request for scenario {request.scenario_number}.")
            worker.b_waiting_for_analyze_scenario_request = False
            worker.run_scenario_submit_time = None # Sometimes the worker was not seen as running the scenario even though it was
            response.received = False
            return response

        worker.analyze_scenario_request = request
        worker.b_waiting_for_analyze_scenario_request = False
        worker.run_scenario_submit_time = None # Sometimes the worker was not seen as running the scenario even though it was
        # worker.b_vehicle_nodes_can_run = False # Is this needed?
        self.log("info", f"Received AnalyzeScenario request: Worker {wid} / Scenario {request.scenario_number}")
        
        # Return response
        response.received = True
        return response

    def register_vehicle_node_service_cb(self, request, response):
        """Service callback for registering a vehicle node"""
        wid = request.worker_id

        if wid not in self.recognized_wids:
            self.log("warn", f"Received RegisterVehicleNode request from unknown Worker {wid}. Ignoring request.")
            response.received = False
            return response

        worker = self.workers[wid]
        if request.node_name not in worker.registered_vehicle_nodes:
            worker.registered_vehicle_nodes.append(request.node_name)
            self.log("info", f"Registered vehicle node {request.node_name} for Worker {wid}")

            if len(worker.registered_vehicle_nodes) == self.num_vehicle_nodes:
                worker.reset()
                self.log("info", f"Worker {wid} vehicle nodes are all registered")
        else:
            self.log("info", f"Vehicle node {request.node_name} is already registered for Worker {wid}")
            
        # Return response
        response.received = True
        return response
    
    def unregister_vehicle_node_service_cb(self, request, response):
        """Service callback for unregistering a vehicle node"""
        wid = request.worker_id

        if wid not in self.recognized_wids:
            self.log("warn", f"Received UnregisterVehicleNode request for vehicle node {request.node_name} from unkown Worker {wid}. Ignoring request.")
            response.received = False
            return response
        
        worker = self.workers[wid]
        if request.node_name not in worker.registered_vehicle_nodes:
            self.log("warn", f"Received UnregisterVehicleNode request to for unknown vehicle node {request.node_name} for Worker {wid}. Ignoring request.")
            response.received = False
            return response

        if request.node_name in worker.registered_vehicle_nodes:
            worker.registered_vehicle_nodes.remove(request.node_name)
            if request.node_name in worker.ready_vehicle_nodes:
                worker.ready_vehicle_nodes.remove(request.node_name)
            worker.b_vehicle_nodes_can_run = False
            worker.b_ignore_analyze_scenario_request = True
            self.log("info", f"Unregistered vehicle node {request.node_name} for Worker {wid}.") # Scenario will get resubmitted when it re-registers

        # Return response
        response.received = True
        return response
    
    def notify_ready_service_cb(self, request, response):
        """Service callback for when a vehicle node notifies it is ready to proceed"""
        wid = request.worker_id

        if wid not in self.recognized_wids:
            self.log("warn", f"Received NotifyReady request from unknown Worker {wid}. Ignoring request.")
            response.received = True
            response.accepted = False
            return response

        worker = self.workers[wid]
        if request.node_name in worker.registered_vehicle_nodes and request.node_name not in worker.ready_vehicle_nodes:
            # Scenario number of 0 is reserved for post vehicle node registration
            if request.last_scenario_number == 0 or request.last_scenario_number == worker.scenario_number:
                worker.ready_vehicle_nodes.append(request.node_name)
                response.accepted = True

                if not worker.b_vehicle_node_requested_rerun and request.request_rerun: # Only store the first rerun request, even if multiple nodes request it
                    worker.b_vehicle_node_requested_rerun = True
                    if "Base Class Issue" not in request.reason_for_rerun:
                        worker.num_rerun_requests += 1
                    self.log("warn", f"Worker {wid} vehicle node {request.node_name} requested rerun with reason: {request.reason_for_rerun}.")
            else:
                self.log("warn", f"Received NotifyReady request from node {request.node_name}. Current scenario number is {worker.scenario_number} but request is for {request.last_scenario_number}. Ignoring request.")
                response.accepted = False
        else:
            self.log("warn", f"Received NotifyReady request from node {request.node_name}. This node is not registered or this is a repeat notification. Ignoring request.")
            response.accepted = False
            
        if self.are_all_vehicle_nodes_ready(wid):
            worker.b_vehicle_nodes_can_run = True

        # Return response
        response.received = True
        return response
    
    def worker_issue_service_cb(self, request, response):
        wid = request.worker_id

        if wid not in self.recognized_wids:
            self.log("warn", f"Received WorkerIssueNotification request from unknown Worker {wid}. Ignoring request.")
            response.received = False
            return response
        
        if request.issue_id == auto_scene_gen_srvs.WorkerIssueNotification.Request.ISSUE_ROSBRIDGE_INTERRUPTED:
            self.log("info", f"Worker {wid} issue notification: ISSUE_ROSBRIDGE_INTERRUPTED. Will re-register worker.")
            self.workers[wid].set_offline(self.WORKER_PROCESS_REQUEST_TIMEOUT)
            response.received = True
            return response
            
        elif request.issue_id == auto_scene_gen_srvs.WorkerIssueNotification.Request.ISSUE_PROBLEM_CREATING_SCENE:
            self.log("warn", f"Worker {wid} issue notification: ISSUE_PROBLEM_CREATING_SCENE. Message provided: '{request.message}'")
            # TODO
            response.received = True
            return response

    def are_all_vehicle_nodes_ready(self, wid: int):
        """Indicate if the appropriate amount of vehicle nodes are registered and ready for a specific worker
        
        Args:
            - wid: Worker ID
        """
        worker = self.workers[wid]
        if len(worker.registered_vehicle_nodes) != self.num_vehicle_nodes:
            return False
        
        for registered_name in worker.registered_vehicle_nodes:
            if registered_name not in worker.ready_vehicle_nodes:
                return False
        return True

    def utility_loop_timer_cb(self):
        """A callback that constantly performs basic utilities, except for the main_loop_wid worker to avoid any conflicts"""
        self.publish_online_status()
        self.publish_vehicle_node_operating_info()
        self.publish_scene_descriptions()
        self.check_run_scenario_response_timer_cb()

    def main_loop_timer_cb(self):
        """This is the main loop that will run until the node is destroyed"""
        if self.b_shutting_down:
            return
        self.main_step(self.main_loop_wid)
        self.main_loop_wid += 1
        if self.main_loop_wid == self.base_wid + self.num_workers:
            self.main_loop_wid = self.base_wid

    def main_step(self, wid: int):
        """In your child class, put your code for running a single step for the given worker here. This function is automatically called by main_loop_timer_cb().
        
        Args:
            - wid: Worker ID being processed
        """
        pass
        
    def shutdown(self):
        """Call this function immediately after rclpy.ok() returns False to properly shutdown this node. 
        Upon shutdown we need to send an offline signal to all AutoSceneGen workers and any other nodes relying on this node's status.
        """
        self.log("info", "-" * 30 + "SHUTDOWN" + "-" * 30, b_log_ros=False)
        self.log("info", f"Sending offline signal")
        self.b_shutting_down = True
        for i in range(10):
            msg = auto_scene_gen_msgs.StatusCode()
            msg.status = auto_scene_gen_msgs.StatusCode.OFFLINE
            self.asg_client_status_pub.publish(msg)
            time.sleep(0.05)