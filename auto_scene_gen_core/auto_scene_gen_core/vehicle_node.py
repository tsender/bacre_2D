import time
import os
import shutil
import datetime
import paramiko
import getpass
import pickle
import traceback
import threading

import rclpy
import rclpy.callback_groups
import rclpy.clock
import rclpy.executors
from rclpy.executors import _rclpy
import rclpy.handle
import rclpy.node
import rclpy.time
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import rosgraph_msgs.msg as rosgraph_msgs
import auto_scene_gen_msgs.msg as auto_scene_gen_msgs
import auto_scene_gen_msgs.srv as auto_scene_gen_srvs

import auto_scene_gen_core.stamp_utils as stamp_utils

class VehicleNodeException(Exception):
    pass


class AutoSceneGenVehicleNode(rclpy.node.Node):
    """This class defines the base functionality for an AutoSceneGen vehicle node. 
    """
    
    DUR_BEFORE_RESUBMITTING = 30.

    def __init__(self, 
                node_name: str
                ):
        super().__init__(node_name)

        # ROS parameters
        self.vehicle_name = self.declare_parameter("vehicle_name", "vehicle").value                 # The name given to the corresponding AutoSceneGenVehicle inside Unreal Engine
        self.wid = self.declare_parameter('wid', 0).value                                           # The AutoSceneGenWorker ID for the associated AutoSceneGenVehicle
        self.asg_client_name = self.declare_parameter("asg_client_name", "asg_client").value        # The name of the AutoSceneGenClient managing the associated worker
        self.asg_client_ip_addr = self.declare_parameter("asg_client_ip_addr", "").value            # The IP address of the computer where the AutoSceneGenClient resides. Leave empty if the client is on the same machine.
        self.ssh_username = self.declare_parameter("ssh_username", "").value                        # The SSH username for sending files between this computer and the computer where the AutoSceneGenClient lives. If empty, then we will use the current username.
        self.b_debug_mode = self.declare_parameter("debug_mode", False).value                       # In case you want to test some of your vehcle's code and want the node to be free from this interface, then set this flag to true. 
                                                                                                    # Note: this is still a bit of an experimental feature, and it may get improved or removed in the future. True = Ignore worker status, False = otherwise.

        self.b_vehicle_enabled = False
        self.b_registered_with_asg_client = False
        self.b_services_online = False
        self.b_ok_to_run = False
        self.last_vehicle_ok_status = False

        self.asg_client_status = auto_scene_gen_msgs.StatusCode.OFFLINE
        self.worker_status = auto_scene_gen_msgs.StatusCode.OFFLINE
        self.register_node_future_response = None
        self.register_node_send_time = None
        self.notify_ready_future_response = None
        self.notify_ready_send_time = None
        self.save_dir = None
        self.b_save_minimal = True
        self.scenario_number = 0
        self.scene_description = None

        # Create an empty temporary save folder
        self.temp_save_dir = os.path.join(os.getcwd(), "vehicle_node_temp", f"worker{self.wid}", self.get_name())
        self.temp_log_filename = os.path.join(self.temp_save_dir, "log.txt")
        if os.path.isdir(self.temp_save_dir):
            shutil.rmtree(self.temp_save_dir)
        os.makedirs(self.temp_save_dir, exist_ok = True) # Recreate directory since above call will delete it

        # ROS clocks
        self.clock_cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.clock_sub = self.create_subscription(rosgraph_msgs.Clock, f'/clock{self.wid}', self.clock_cb, 1, callback_group=self.clock_cb_group)
        self.sim_clock = rclpy.clock.ROSClock()
        self.sim_clock._set_ros_time_is_active(True)
        self.system_clock = rclpy.clock.Clock()

        # ROS subs
        self.worker_topic_prefix = f"/asg_worker{self.wid}"
        self.vehicle_topic_prefix = f"/asg_worker{self.wid}/{self.vehicle_name}"
        
        self.asg_client_status_sub = self.create_subscription(auto_scene_gen_msgs.StatusCode, f'/{self.asg_client_name}/status', self.asg_client_status_cb, 10)
        self.worker_status_sub = self.create_subscription(auto_scene_gen_msgs.StatusCode, f"/asg_worker{self.wid}/status", self.worker_status_cb, 10)
        self.vehicle_status_sub = self.create_subscription(auto_scene_gen_msgs.VehicleStatus, self.vehicle_topic_prefix + "/status", self.vehicle_status_cb, 10)
        self.vehicle_node_operating_info_sub = self.create_subscription(auto_scene_gen_msgs.VehicleNodeOperatingInfo, f'/{self.asg_client_name}/vehicle_node_operating_info', self.vehicle_node_operating_info_cb, 10)
        self.scene_description_sub = self.create_subscription(auto_scene_gen_msgs.SceneDescription, f"/asg_worker{self.wid}/scene_description", self.scene_description_cb, 10)

        # ROS clients
        self.register_node_cli = self.create_client(auto_scene_gen_srvs.RegisterVehicleNode, f"/{self.asg_client_name}/services/register_vehicle_node")
        self.unregister_node_cli = self.create_client(auto_scene_gen_srvs.RegisterVehicleNode, f"/{self.asg_client_name}/services/unregister_vehicle_node")
        self.notify_ready_cli = self.create_client(auto_scene_gen_srvs.NotifyReady, f"/{self.asg_client_name}/services/notify_ready")

        # ROS timers
        self.register_node_timer = self.create_timer(1., self.register_node_timer_cb, clock=self.system_clock)

        self.notify_ready_request = auto_scene_gen_srvs.NotifyReady.Request()
        self.notify_ready_request.worker_id = self.wid
        self.notify_ready_request.node_name = self.get_name()
        self.notify_ready_request.last_scenario_number = 0
        self.notify_ready_request.request_rerun = False
        self.notify_ready_request.reason_for_rerun = ""

        self.vehicle_enabled_timestamp = None
        self.vehicle_disabled_timestamp = None

        # Threading event
        self.vehicle_disabled_event = threading.Event()

        # Configure SSH settings
        self.b_use_ssh = False
        self.ssh_private_key = None
        self._find_ssh_private_key()

        self.log('info', f'Initialized AutoSceneGenVehicleNode for Worker {self.wid}')

    def log(self, log_level: str, msg: str, b_write_to_file: bool = True):
        """Log information via ROS and possibly write to a log file.
        
        Args:
            - log_level: Log level (info, warn, error)
            - msg: The string to log via ROS and to append to the log file (if requested)
            - b_write_to_file: Indicates if we should write to the temporary log file, if it exists (will eventually be copied to the actual save directory)
        
        """
        if str.lower(log_level) == 'info':
            self.get_logger().info(f"[{datetime.datetime.now()}] " + msg)
        elif str.lower(log_level) == 'warn':
            self.get_logger().warn(f"[{datetime.datetime.now()}] " + msg)
        elif str.lower(log_level) == 'error':
            self.get_logger().error(f"[{datetime.datetime.now()}] " + msg)
        else:
            self.get_logger().info(f"[{datetime.datetime.now()}] " + msg)

        # if b_write_to_file and self.save_dir is not None and self.temp_log_filename is not None:
        if b_write_to_file and os.path.isdir(self.temp_save_dir):
            # Log message to log file with date and time
            with open(self.temp_log_filename, "a") as f:
                if str.lower(log_level) == 'info':
                    log_file_msg = f"[{datetime.datetime.now()}] [INFO] " + msg
                elif str.lower(log_level) == 'warn':
                    log_file_msg = f"[{datetime.datetime.now()}] [WARN] " + msg
                elif str.lower(log_level) == 'error':
                    log_file_msg = f"[{datetime.datetime.now()}] [ERROR] " + msg
                else:
                    log_file_msg = f"[{datetime.datetime.now()}] [INFO] " + msg
                
                f.write(log_file_msg + "\n")
    
    def wait_for_services(self):
        """Blocking call, waits for client to come online and then waits for services to come online."""
        self.log('info', "Waiting for services...")
        while not self.register_node_cli.wait_for_service(timeout_sec=1.0):
            pass
        while not self.notify_ready_cli.wait_for_service(timeout_sec=1.0):
            pass
        self.b_services_online = True
        self.log("info", f"Client is online")

    def _get_ssh_client(self, ip_addr: str, username: str, private_key: paramiko.PKey, b_log_errors: bool = True):
        """Get an ssh client from the speciifed parameters

        Args:
            - ip_addr: Remote IP address
            - username: Username to log into remote system
            - private_key: The private key needed for authentication
            - b_log_errors: Indicate if we should log errors to ROS

        Returns:
            - The ssh_client or None (if unable to form a connection)
        """
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh_client.connect(ip_addr, username=username, pkey = private_key)
            return ssh_client
        except Exception:
            if b_log_errors:
                self.log("error", f"Exception when connecting user '{username}' to SSH client at IP '{ip_addr}': {traceback.format_exc()}")
            ssh_client.close()
            return None

    def _find_ssh_private_key(self):
        """Find a valid ssh private key. WIll use the first valid one we find, it it exists."""

        if self.asg_client_ip_addr == "":
            self.b_use_ssh = False
            return            

        self.log("info", "Searching for ssh private key...")
        if self.ssh_username == "":
            username = getpass.getuser()
        else:
            username = self.ssh_username

        if os.path.exists(os.path.expanduser("~/.ssh/id_dss")):
            pkey = paramiko.DSSKey.from_private_key_file(os.path.expanduser("~/.ssh/id_dss"))
            ssh_client = self._get_ssh_client(self.asg_client_ip_addr, username, pkey, b_log_errors=False)
            if ssh_client is not None:
                ssh_client.close()
                self.ssh_private_key = pkey
                self.b_use_ssh = True
                self.log("info", f"Found valid DSS ssh private key.")
                return
            
        if os.path.exists(os.path.expanduser("~/.ssh/id_rsa")):
            pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser("~/.ssh/id_rsa"))
            ssh_client = self._get_ssh_client(self.asg_client_ip_addr, username, pkey, b_log_errors=False)
            if ssh_client is not None:
                ssh_client.close()
                self.ssh_private_key = pkey
                self.b_use_ssh = True
                self.log("info", f"Found valid RSA ssh private key.")
                return
            
        if os.path.exists(os.path.expanduser("~/.ssh/id_ecdsa")):
            pkey = paramiko.ECDSAKey.from_private_key_file(os.path.expanduser("~/.ssh/id_ecdsa"))
            ssh_client = self._get_ssh_client(self.asg_client_ip_addr, username, pkey, b_log_errors=False)
            if ssh_client is not None:
                ssh_client.close()
                self.ssh_private_key = pkey
                self.b_use_ssh = True
                self.log("info", f"Found valid ECDSA ssh private key.")
                return
            
        if os.path.exists(os.path.expanduser("~/.ssh/id_ed25519")):
            pkey = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser("~/.ssh/id_ed25519"))
            ssh_client = self._get_ssh_client(self.asg_client_ip_addr, username, pkey, b_log_errors=False)
            if ssh_client is not None:
                ssh_client.close()
                self.ssh_private_key = pkey
                self.b_use_ssh = True
                self.log("info", f"Found valid ED25519 ssh private key.")
                return
                
        self.log("warn", f"Could not find a valid private ssh key. Cannot use ssh to transfer files remotely.")
        self.ssh_private_key = None
        self.b_use_ssh = False

    def save_file_on_remote_asg_client(self, local_path: str, remote_path: str):
        """Save a file on the remote AutoSceneGenClient. The file is assumed to have already been saved locally.
        
        Args:
            - local_path: The local file to save on the remote
            - remote_path: The remote path to write to on the AutoSceneGenClient machine

        Returns:
            True if successful, False otherwise
        """
        self.log("info", f"Saving file to remote AutoSceneGenClient with remote file path: {remote_path}")
        if not self.b_use_ssh or not os.path.exists(local_path):
            self.log("warn", f"Cannot use ssh or local file path '{local_path}' does not exist.")
            return False
        
        if self.ssh_username == "":
            username = getpass.getuser()
        else:
            username = self.ssh_username
        ssh_client = self._get_ssh_client(self.asg_client_ip_addr, username, self.ssh_private_key)
        if ssh_client is None:
            return False
        
        try:
            sftp_client = ssh_client.open_sftp()
            sftp_client.put(local_path, remote_path)
            sftp_client.close()
            ssh_client.close()
            return True
        except Exception:
            self.log("error", f"Exception when trying to save local file with path '{local_path}' to SSH client: {traceback.format_exc()}")
            return False
        
    def copy_file_from_remote_asg_client(self, remote_path: str, local_path: str, timeout: float = 300.):
        """Copy a file from the remote AutoSceneGenClient. Will timeout after a set duration.
        
        Args:
            - remote_path: The filename to look for on the remote AutoSceneGenClient.
            - local_path: The filename to save to on the local machine.
            - timeout: Maximum waiting period in [s] to look for the file. Set to None to disable.

        Returns:
            True if successful, False otherwise
        """
        self.log("info", f"Copying file from remote AutoSceneGenClient with remote file path: {remote_path}")
        if not self.b_use_ssh:
            self.log("warn", f"Cannot use ssh.")
            return False
        
        if self.ssh_username == "":
            username = getpass.getuser()
        else:
            username = self.ssh_username
        ssh_client = self._get_ssh_client(self.asg_client_ip_addr, username, self.ssh_private_key)
        if ssh_client is None:
            return False

        b_file_found = False
        retry_delay = 1.
        start_time = time.time()
        sftp_client = ssh_client.open_sftp()
        while not b_file_found:
            try:
                sftp_client.get(remote_path, local_path)
                b_file_found = True
                self.log("info", f"Found remote file and saved it locally to: {local_path}.")
                sftp_client.close()
                ssh_client.close()
                return True
            except Exception: # The only exception should be a FileNotFoundError
                if timeout is not None and (time.time() - start_time) >= timeout:
                    self.log("warn", f"Failed to retrieve remote file due to timeout.")
                    sftp_client.close()
                    ssh_client.close()
                    return False
                time.sleep(retry_delay)
    
    def save_pickle_object_on_asg_client(self, data, filename: str):
        """Save a pickle object to the AutoSceneGenClient (will determine if the client is local or remote)
        
        Args:
            - data: The data object to serialize with pickle
            - filename: The filename of the object (this is NOT the file path)
        """
        if self.save_dir is None:
            return
        
        if self.b_use_ssh:
            local_path = os.path.join(self.temp_save_dir, filename)
            remote_path = os.path.join(self.save_dir, filename)
            with open(local_path, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            self.save_file_on_remote_asg_client(local_path, remote_path)
        else:
            local_path = os.path.join(self.save_dir, filename)
            with open(local_path, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            self.log("info", f"Saved pickle object to AutoSceneGenClient locally: {local_path}")

    def load_pickle_object_from_asg_client(self, filename: str, timeout: float = 300.):
        """Load a pickle object from the AutoSceneGenClient (will determine if the client is local or remote)
        
        Args:
            - filename: The filename of the object (this is NOT the file path)
            - timeout: Maximum waiting period in [s] to look for the file. Set to None to disable.

        Returns:
            The unpickeled object, if the file was found, otherwise returns None.
        """
        if self.b_use_ssh:
            local_path = os.path.join(self.temp_save_dir, filename)
            remote_path = os.path.join(self.save_dir, filename)
            if self.copy_file_from_remote_asg_client(remote_path, local_path, timeout=timeout):
                with open(local_path, 'rb') as f:
                    data = pickle.load(f)
                    self.log("info", f"Loaded pickle object from AutoSceneGenClient remotely from path: {remote_path}")
                return data
            else:
                return None
        else:
            local_path = os.path.join(self.save_dir, filename)
            start_time = time.time()
            while not os.path.exists(local_path):
                time.sleep(1.0)
                if timeout is not None and (time.time() - start_time) >= timeout:
                    self.log("warn", f"Failed to retrieve local file due to timeout.")
                    return None
            with open(local_path, 'rb') as f:
                data = pickle.load(f)
                self.log("info", f"Loaded pickle object from AutoSceneGenClient locally from path: {local_path}")
            return data

    def clock_cb(self, msg: rosgraph_msgs.Clock):
        self.sim_clock.set_ros_time_override(rclpy.time.Time(seconds=msg.clock.sec, nanoseconds=msg.clock.nanosec, clock_type=rclpy.clock.ClockType.ROS_TIME))

    def save_node_data(self):
        """Save any internal data from the node. Override this function in the child class."""
        pass

    def reset(self):
        """Reset any internal variables. Override this function in the child class."""
        pass
    
    def check_for_rerun_request(self):
        """Vehicle nodes can implement this function as a means to check if they need to request a rerun 
        (e.g., in case they encountered a problem during the simulation that would affect the results)

        If a rerun is requested, overwrite the self.notify_ready_request.request_rerun and self.notify_ready_request.reason_for_rerun fields.
        """
        pass

    def asg_client_status_cb(self, msg: auto_scene_gen_msgs.StatusCode):
        """Keep track of the client's status"""
        self.asg_client_status = msg.status
        if self.asg_client_status == auto_scene_gen_msgs.StatusCode.OFFLINE:
            self.log("info", f"Client went offline")
            self.b_registered_with_asg_client = False
            self.b_services_online = False
            self.register_node_future_response = None
            self.b_ok_to_run = False

    def worker_status_cb(self, msg: auto_scene_gen_msgs.StatusCode):
        """Keep track of the worker's status"""
        self.worker_status = msg.status

    def vehicle_status_cb(self, msg: auto_scene_gen_msgs.VehicleStatus):
        """Keep track of the vehicle's status. When the vehicle is disabled, go through the reset process 
        (save any data, check for rerun request, reset any internal variables, etc.).
        """
        if self.b_debug_mode:
            self.b_vehicle_enabled = msg.enabled
        else:
            self.b_vehicle_enabled = msg.enabled and self.worker_status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_RUNNING

        if not self.vehicle_ok() and self.last_vehicle_ok_status:
            self.vehicle_disabled_timestamp = self.sim_clock.now().to_msg()
            self.vehicle_disabled_event.set()
            self.log('info', f"Vehicle is disabled.")

            if self.b_registered_with_asg_client and not msg.preempted:
                b_saved_node_data = True
                if self.save_dir is not None:
                    try:
                        self.save_node_data()
                    except Exception:
                        b_saved_node_data = False
                        self.notify_ready_request.request_rerun = True
                        self.notify_ready_request.reason_for_rerun = f"Base Class Issue: Encountered exception when saving node data: {traceback.format_exc()}"
                        self.log("warn", f"Requesting rerun with reason: {self.notify_ready_request.reason_for_rerun}")

                if b_saved_node_data:
                    self.check_for_rerun_request()

                # Save log file on client computer
                if self.save_dir is not None:
                    remote_log_filename = os.path.join(self.save_dir, f"vehicle_node_log_{self.get_name()}.txt")
                    if os.path.exists(self.temp_log_filename):
                        if self.b_use_ssh:
                            self.save_file_on_remote_asg_client(self.temp_log_filename, remote_log_filename)
                        else:
                            shutil.copy(self.temp_log_filename, remote_log_filename)

            # Empty contents in temporary save directory. Do this here, so if there is a problem resetting, we can still see the error
            if os.path.isdir(self.temp_save_dir):
                shutil.rmtree(self.temp_save_dir)
            os.makedirs(self.temp_save_dir, exist_ok = True) # Recreate directory since above call will delete it
            
            self.reset()
            
            if not msg.preempted:
                self.notify_ready_request.last_scenario_number = self.scenario_number
                self.send_ready_notification()

        if self.vehicle_ok() and not self.last_vehicle_ok_status:
            self.vehicle_enabled_timestamp = self.sim_clock.now().to_msg()
            self.vehicle_disabled_event.clear()
            self.log('info', f"Vehicle is enabled and ok.")

        self.last_vehicle_ok_status = self.vehicle_ok()

    def vehicle_node_operating_info_cb(self, msg: auto_scene_gen_msgs.VehicleNodeOperatingInfo):
        """Keep track of important operating information"""
        if self.wid in msg.worker_ids:
            idx = msg.worker_ids.index(self.wid)
            self.scenario_number = msg.scenario_numbers[idx]
            self.b_ok_to_run = msg.ok_to_run[idx]
            self.save_dir = msg.save_dirs[idx]
            self.b_save_minimal = msg.save_minimal[idx]

    def scene_description_cb(self, msg: auto_scene_gen_msgs.SceneDescription):
        """Keep track of the current scene description being run on the worker"""
        self.scene_description = msg

    def register_node_with_asg_client(self, b_resending: bool = False):
        """Submit request to register vehicle node with client"""
        if not self.b_debug_mode and self.asg_client_status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_RUNNING:
            req = auto_scene_gen_srvs.RegisterVehicleNode.Request()
            req.worker_id = self.wid
            req.node_name = self.get_name()
            self.register_node_send_time = time.time()
            self.register_node_future_response = self.register_node_cli.call_async(req)

            if b_resending:
                self.log("info", "Resending RegisterVehicleNode request")
            else:
                self.log("info", "Sending RegisterVehicleNode request")

    def send_ready_notification(self, b_resending: bool = False):
        """Send ready notification to the client"""
        if not self.b_debug_mode and self.b_registered_with_asg_client:
            self.notify_ready_send_time = time.time()
            self.notify_ready_future_response = self.notify_ready_cli.call_async(self.notify_ready_request)

            if b_resending:
                self.log("info", f"Resending NotifyReady request for scenario {self.notify_ready_request.last_scenario_number}")
            else:
                self.log("info", f"Sending NotifyReady request for scenario {self.notify_ready_request.last_scenario_number}")

    def register_node_timer_cb(self):
        """Make sure this node is registered with the client, and make sure NotifyReady requests are received"""
        if self.b_debug_mode:
            return

        if not self.b_registered_with_asg_client and not self.b_services_online:
            self.wait_for_services()

        if not self.b_registered_with_asg_client and self.register_node_future_response is None:
            self.register_node_with_asg_client()

        # Make sure node is registered
        if not self.b_registered_with_asg_client and self.register_node_future_response is not None:
            if self.register_node_future_response.done():
                try:
                    response = self.register_node_future_response.result()
                except Exception as e:
                    self.log('error', f"Registering vehicle node failed with exception: {e}")
                    self.register_node_with_asg_client() # Try again?
                else:
                    if response.received:
                        self.b_registered_with_asg_client = True
                        self.register_node_future_response = None
                        self.register_node_send_time = None
                        self.notify_ready_request.last_scenario_number = 0
                        self.send_ready_notification()
                    else:
                        self.register_node_with_asg_client(b_resending=True) # Send again

        now = time.time()
        if not self.b_registered_with_asg_client and self.register_node_send_time is not None:
            if now - self.register_node_send_time > self.DUR_BEFORE_RESUBMITTING:
                self.register_node_with_asg_client(b_resending=True)
        
        # Make sure ready notification was received
        if self.b_registered_with_asg_client and self.notify_ready_future_response is not None:
            if self.notify_ready_future_response.done():
                try:
                    response = self.notify_ready_future_response.result()
                except Exception as e:
                    self.log('error', f"NotifyReady request failed with exception: {e}")
                    self.send_ready_notification(b_resending=True) # Send again
                else:
                    if response.received:
                        # Reset these two fields
                        self.notify_ready_request.request_rerun = False
                        self.notify_ready_request.reason_for_rerun = ""

                        self.notify_ready_future_response = None
                        self.notify_ready_send_time = None
                        self.log("info", f"NotifyReady response for scenario {self.notify_ready_request.last_scenario_number} was received with acceptance {response.accepted}")
                    else:
                        self.send_ready_notification(b_resending=True) # Send again
        
        now = time.time()
        if self.b_registered_with_asg_client and self.notify_ready_send_time is not None:
            if now - self.notify_ready_send_time > self.DUR_BEFORE_RESUBMITTING:
                self.send_ready_notification(b_resending=True)

    def vehicle_ok(self):
        """Indicates if the vehicle is enabled and if this node is registered with the client."""
        if self.b_debug_mode:
            return self.b_vehicle_enabled
        else:
            return self.b_registered_with_asg_client and self.notify_ready_send_time is None and self.b_vehicle_enabled and self.b_ok_to_run
        
    def shutdown(self):
        """Call this function immediately after the node dies (but before the main function terminates)"""
        if not self.b_debug_mode and self.asg_client_status == auto_scene_gen_msgs.StatusCode.ONLINE_AND_RUNNING:
            self.log("info", "Unregistering vehicle node.")
            req = auto_scene_gen_srvs.RegisterVehicleNode.Request()
            req.worker_id = self.wid
            req.node_name = self.get_name()
            response = self.unregister_node_cli.call_async(req)
            time.sleep(0.1)


def spin_vehicle_node(node: AutoSceneGenVehicleNode, num_threads: int = None):
    """Spin a given AutoSceneGenVehicleNode. 
    This function will ignore RCLError and InvalidHandle Exceptions, since they are usually a side-effect of the reset process
    
    Args:
        - node: The node to spin
        - num_threads: Number of threads to use with the multithreaded executor
    """
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads) # Must use multithreaded executor so we can update the simulation clock time
    executor.add_node(node)
    try:
        while executor.context.ok():
            try:
                executor.spin_once()
            except _rclpy.RCLError:
                node.log('warn', f"Ignoring RCLError: {traceback.format_exc()}")
            except rclpy.handle.InvalidHandle:
                node.log('warn', f"Ignoring InvalidHandle: {traceback.format_exc()}")
    except KeyboardInterrupt:
        node.log("info", "Keyboard interrupt. Shutting down...")
    except Exception as e:
        node.log('error', f"EXCEPTION: {traceback.format_exc()}") # Print entire exception and source of problem
    finally:
        node.shutdown()
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()