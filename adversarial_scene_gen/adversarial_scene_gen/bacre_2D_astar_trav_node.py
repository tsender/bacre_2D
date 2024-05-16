import math
import os
import shutil
import time
import io
import cv2
import glob
import numpy as np
import traceback
import pickle
import copy
import multiprocessing
import transforms3d
from typing import Tuple, List, Dict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import rclpy
import rclpy.node
import std_msgs.msg as std_msgs

import auto_scene_gen_msgs.msg as auto_scene_gen_msgs
import auto_scene_gen_msgs.srv as auto_scene_gen_srvs

from auto_scene_gen_core.client_node import AutoSceneGenWorkerRef, AutoSceneGenClient
from auto_scene_gen_core.scenario_builder import AutoSceneGenScenarioBuilder, StructuralSceneActorAttributes, TexturalAttributes, OperationalAttributes, StructuralSceneActorConfig
import auto_scene_gen_core.ros_image_converter as ros_img_converter
import auto_scene_gen_core.stamp_utils as stamp_utils

from astar_trav.astar_voting_path_planner import AstarVotingPathPlanner
from astar_trav.astar_trav_2D_all_in_one_node import AstarPathSnapshot, AstarSnapshotsSummary
from astar_trav.simple_path_follower_node import PathFollowerTrackingSnapshot, PathFollowerSnapshotsSummary


def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))

def linear_interp(x1, x2, r):
    """Linear interpolation from x1 to x2 with fraction r"""
    return x1 + r * (x2 - x1)

def cross_linear_interp(x, vals, out_arr):
    """Cross linear interpolation. Output a value interpolated from the out_arr array based on the location of x in the vals array.
    
    Args:
        - x: Reference search value
        - vals: A list/array of monotonically increasing elements. The location of x in this array determines where to interpolate from in out_arr.
        - out_arr: The list/array from which the output will be interpolated from. It is assumed that vals[i] pairs with out_arr[i].

    Returns:
        - The interpolated value, or None if no interpolation can be performed.
    """
    for i in range(len(vals)-1):
        if x == vals[i]:
            return out_arr[i]
        
        if x == vals[i+1]:
            return out_arr[i+1]
    
        if vals[i] < x < vals[i+1]:
            delta = (x - vals[i]) / (vals[i+1] - vals[i])
            return out_arr[i] + delta * (out_arr[i+1] - out_arr[i])
    return None # In case of some error, return None

def get_smoothing_avg(array: List, moving_avg_len):
    smoothed_array = []
    for i in range(len(array)):
        start = max(0, i + 1 - moving_avg_len)
        smoothed_array.append(np.mean(array[start:i+1]))
    return smoothed_array

def binary_search_dec_idx(arr, value, key = lambda x: x):
    """Get the insertion index to place value in a sorted decreasing array
    
    Args:
        - arr: Input array/list
        - value: New value to insert
        - key: Optional function used for comparing elements
    
    Returns:
        - The insertion index such that all elements to the left are strictly greater in value
    """
    if len(arr) == 0:
        return 0
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = low + math.ceil((high - low)/2) # Reduces potential overflow from using (low + high) // 2
        if key(arr[mid]) > key(value):
            low = mid + 1
        elif key(value) > key(arr[mid]):
            high = mid -1
        else:
            return mid
    return low # Because it's unlikely an existing element has the same value, the correct insertion index is actually low

def binary_search_inc_idx(arr, value, key = lambda x: x):
    """Get the insertion index to place value in a sorted increasing array
    
    Args:
        - arr: Input array/list
        - value: New value to insert
        - key: Optional function used for comparing elements
    
    Returns:
        - The insertion index such that all elements to the left are strictly less in value
    """
    if len(arr) == 0:
        return 0
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = low + math.ceil((high - low)/2)
        if key(arr[mid]) < key(value):
            low = mid + 1
        elif key(value) < key(arr[mid]):
            high = mid -1
        else:
            return mid
    
    return low


class Regret:
    """Object for storing all regret components"""
    def __init__(self):
        self.value = None # Actual regret value
        self.cost_error = None
        self.time_error = None
        self.equiv_cost_pos_error = None
        self.equiv_time_pos_error = None

    def set_to_zero(self):
        self.value = 0.
        self.cost_error = 0.
        self.time_error = 0.
        self.equiv_cost_pos_error = 0.
        self.equiv_time_pos_error = 0.


class ScenarioOutcome:
    """Contains information regarding the outcome of a single scenario run/trial"""
    def __init__(self):
        """Initialize a scenario outcome"""
        self.termination_reason = -1 # Reason for terminating the scenario (e.g., success, crash, etc.)
        self.succeeded = False # Indicates if vehicle reached the goal
        self.sim_time = None # Total simulation time for the vehicle
        self.vehicle_path = None # Numpy array, each row is a (x,y,z) pair from the discetized vehicle path in [m]
        self.yaw_angles = None
        self.vehicle_orientation = None # List of (w,x,y,z) quaternions
        self.regret = Regret() # The vehicle's regret for this particular scenario run

        # Vehicle node data
        self.astar_path_snapshots : List[AstarPathSnapshot] = None
        self.astar_snapshots_summary : AstarSnapshotsSummary = None
        self.path_follower_snapshots : List[PathFollowerTrackingSnapshot] = None
        self.path_follower_snapshots_summary : PathFollowerSnapshotsSummary = None


class Scenario:
    """Keeps track of useful data about a given scenario. Stats about the scenario should be based on the average of the outcomes, as
    running a single trial is not enough to determine how well a vehicle will perform in that scenario (due to non-determinism and consistency issues in simulation).
    """
    def __init__(self, temp_id: int, parent_id: int, tree_level: int, scene_description: auto_scene_gen_msgs.SceneDescription):
        """Create a scenario object
        
        Args:
            - temp_id: Temporary ID given to this scenario (used for logging purposes)
            - parent_id: Scenario ID of parent
            - tree_level: The level in the tree that this scenario corresponds to. Root level (parent is None) = 0
            - scene_description: Scene description message. It is assumed the start and goal location are known.
        """
        self.id = None # Actual scenario ID, get's set later
        self.temp_id = temp_id # Temporary ID, for logging
        self.parent_id = parent_id # Parent scenario's ID
        self.tree_level = tree_level # The level in the tree that this scenario corresponds to. Root level (parent is None) = 0
        self.scene_description = scene_description # Scene description message
        self.opt_path = None                # (N,2) numpy arrays of path points, each row is a (x,y) point
        self.opt_path_len = None
        self.opt_path_cost = None
        self.opt_path_cost_values = None    # List of N values, each being the total cost accrued at that point
        self.opt_path_time = None
        self.opt_path_time_values = None    # List of N values, each being the total time/duration accrued at that point

        self.outcomes : List[ScenarioOutcome] = []
        self.avg_regret = Regret() # The estimated regret for the vehicle in this particular scenario
        
    def get_vehicle_paths(self):
        """Returns a list of all vehicle paths. In each path, the rows denote the (x,y,z) pairs."""
        paths = []
        for outcome in self.outcomes:
            paths.append(outcome.vehicle_path)
        return paths

    def estimate_regret(self):
        """Estimate the regret for the scenario by averaging the regret over all of the outcomes"""
        n = len(self.outcomes)
        self.avg_regret.set_to_zero()

        for outcome in self.outcomes:
            self.avg_regret.value += outcome.regret.value / n
            self.avg_regret.cost_error += outcome.regret.cost_error / n
            self.avg_regret.time_error += outcome.regret.time_error / n
            self.avg_regret.equiv_cost_pos_error += outcome.regret.equiv_cost_pos_error / n
            self.avg_regret.equiv_time_pos_error += outcome.regret.equiv_time_pos_error / n

    def get_outcome_distribution(self):
        """Return the number of outcomes that the vehicle succeeded and failed in
        
        Returns:
            - Number of succesful vehicle attempts
            - Number of failed vehicle attempts
        """
        nsuccess = 0
        nfail = 0

        for outcome in self.outcomes:
            if outcome.succeeded:
                nsuccess += 1
            else:
                nfail += 1
        return nsuccess, nfail
    

class ScenarioBuilderAndRefAgent(AutoSceneGenScenarioBuilder):
    def __init__(self,
                landscape_nominal_size: float,                   # The side-length of the landscape in [m], this is a square landscape
                landscape_subdivisions: int,                    # The number of times the two base triangles in the nominal landscape should be subdivided
                landscape_border: float,                        # Denotes the approximate length to extend the nominal landscape in [m]
                ssa_attr: StructuralSceneActorAttributes,
                ssa_config: Tuple[Dict],                          # Information about the SSAs. Tuple of dicts with keys: 'path_name',  'num_instances', 'max_scale'
                b_ssa_casts_shadow: bool,                       # Indicates if the SSAs cast a shadow in the game
                b_allow_collisions: bool,                       # Indicate if simulation keeps running in the case of vehicle collisions
                txt_attr: TexturalAttributes,
                opr_attr: OperationalAttributes,
                start_obstacle_free_radius: float,              # Obstacle-free radius [m] around the start location
                goal_obstacle_free_radius: float,              # Obstacle-free radius [m] around the goal location
                astar_path_planner_dict: Dict,                  # Dictionary of params for the A* voting path planner (used to determine global optimal trajecory)
                nominal_vehicle_speed: float,                   # Nominal vehicle speed [m/s]
                opt_agent_min_obstacle_proximity: float,        # The minimum distance the optimal agent can come to any one obstacle
                ):
        super().__init__(
            landscape_nominal_size,  
            landscape_subdivisions, 
            landscape_border,
            ssa_attr,
            ssa_config,       
            b_ssa_casts_shadow,       
            b_allow_collisions,      
            txt_attr,
            opr_attr,
            start_obstacle_free_radius,    
            goal_obstacle_free_radius,
        )
        self.astar_path_planner_dict = astar_path_planner_dict
        self.astar_path_planner = AstarVotingPathPlanner(**astar_path_planner_dict)
        self.nominal_vehicle_speed = nominal_vehicle_speed
        self.opt_agent_min_obstacle_proximity = opt_agent_min_obstacle_proximity

    def log_parameters_to_file(self, file_path: str):
        """Append all parameter data to a provided txt file. Used for logging purposes.
        
        Args:
            - file: The .txt file to append data to
        """
        super().log_parameters_to_file(file_path)
        with open(file_path, "a") as f:
            f.write(f"- astar_path_planner_dict:\n")
            for key,value in self.astar_path_planner_dict.items():
                f.write(f" "*5 + f"- {key}: {value}\n")

            f.write(f"- nominal_vehicle_speed: {self.nominal_vehicle_speed}\n")
            f.write(f"- opt_agent_min_obstacle_proximity: {self.opt_agent_min_obstacle_proximity}\n")

    def configure_reference_agent(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        """Configure the reference agent with the provided RunScenario request

        Args:
            - scenario_request: The input RunScenario request
        """
        self.astar_path_planner.clear_map()
        visible_obs_xy = self.get_visible_ssa_locations(scenario_request.scene_description.ssa_array)
        for obs_xy in visible_obs_xy:
            self.astar_path_planner.add_obstacle_vote(obs_xy, votes_to_add = 50)
        self.astar_path_planner.update_compact_numpy_arrays()

    def get_optimal_path(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        """Compute the optimal trajectory from start to goal using full knowledge of the scene configuration
        
        Args:
            - scenario_request: The input RunScenario request

        Returns:
            - opt_path: Optimal path as list of (x,y) numpy arrays
            - opt_path_length: Arc length of the optimal path
            - opt_cost: Cost of the optimal path
            - opt_path_cost_values: The accrued path cost at each point as a list of floats
        """
        start_location = np.array([scenario_request.vehicle_start_location.x, scenario_request.vehicle_start_location.y])
        goal_location = np.array([scenario_request.vehicle_goal_location.x, scenario_request.vehicle_goal_location.y])
        start_yaw = scenario_request.vehicle_start_yaw * math.pi/180.

        self.configure_reference_agent(scenario_request)
        opt_path, opt_cost, iterations = self.astar_path_planner.construct_optimal_path(start_location, goal_location, start_yaw)
        opt_path_length = self.astar_path_planner.compute_path_length(opt_path)
        opt_edge_cost_breakdown, _ = self.astar_path_planner.get_edge_cost_breakdown(opt_path, start_yaw)
        opt_edge_cost_breakdown.insert(0, 0.)
        opt_path_cost_values = np.cumsum(opt_edge_cost_breakdown).tolist()
        opt_path_cost_values[-1] = opt_cost # Explicitly set this in case there are precision errors
        return opt_path, opt_path_length, opt_cost, opt_path_cost_values

    def is_scenario_request_feasible(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        # Are all obstacles outside of obstacle-free radii?
        start_location = self.opr_attr.get_default_start_location()
        goal_location = self.opr_attr.get_default_goal_location()
        visible_obs_xy = np.array(self.get_visible_ssa_locations(scenario_request.scene_description.ssa_array)) # (n,2) array
        dist_to_start = fast_norm(visible_obs_xy - start_location, axis=-1)
        dist_to_goal = fast_norm(visible_obs_xy - goal_location, axis=-1)
        if np.count_nonzero(dist_to_start <= self.start_obstacle_free_radius):
            return False
        if np.count_nonzero(dist_to_goal <= self.goal_obstacle_free_radius):
            return False
        
        # Can optimal agent finish within sim_timeout_period seconds?
        opt_path, opt_path_len, opt_cost, opt_path_cost_values = self.get_optimal_path(scenario_request)
        opt_sim_time = opt_path_len / self.nominal_vehicle_speed
        if opt_sim_time > scenario_request.sim_timeout_period:
            return False

        # Will the optimal agent get too close to any one obstacle?
        visible_obs_xy = np.array(self.get_visible_ssa_locations(scenario_request.scene_description.ssa_array), dtype=np.float32)
        if visible_obs_xy.size == 0:
            return True
        else:
            opt_path_np = np.array(opt_path, dtype=np.float32).reshape((len(opt_path), 1, 2)) # Each major element is a (1,2) array for an (x,y) position (this let's us do tiling)
            distances = fast_norm(opt_path_np - visible_obs_xy, axis=-1) # Distance from every path point to every obstacle
            if np.min(distances) < self.opt_agent_min_obstacle_proximity:
                return False

        return True


class BACREWorkerRef(AutoSceneGenWorkerRef):
    """Extension of the AutoSceneGenWorkerRef class for BACRE."""
    def __init__(self, wid: int):
        """Initialize an BACREWorkerRef
        
        Args:
            -wid: Worker ID
        """
        super().__init__(wid)
        self.scenario : Scenario = None
        self.num_scenario_runs = 0
        self.b_need_new_scenario = True
        self.b_running_base_scenario = False

    def reset(self):
        self.scenario = None
        self.run_scenario_request = None
        self.analyze_scenario_request = None
        self.num_scenario_runs = 0
        self.b_need_new_scenario = True
        self.b_running_base_scenario = False
        self.num_resubmission_attempts = 0

    def add_scenario_outcome(self, outcome: ScenarioOutcome):
        """Add a scenario outcome. The provided outcome should have most of its fields already populated."""
        outcome.termination_reason = self.analyze_scenario_request.termination_reason
        outcome.succeeded = self.analyze_scenario_request.termination_reason == auto_scene_gen_srvs.AnalyzeScenario.Request.REASON_SUCCESS
        self.scenario.outcomes.append(outcome)
        self.num_scenario_runs = len(self.scenario.outcomes)

    def remove_last_scenario_outcome(self):
        """Removes the last scenario outcome and decrements counter. Returns True if successful, False otherwise."""
        if self.num_scenario_runs > 0:
            self.scenario.outcomes.pop(-1)
            self.num_scenario_runs -= 1
            return True
        else:
            return False

    def get_latest_vehicle_paths(self):
        return self.scenario.get_vehicle_paths()


class ScenarioRegretManager:
    """Helper class to store scenario regrets in one place."""
    def __init__(self):
        self.scenario_regrets : Dict[int, float] = {} # Key = scenario ID
        self.sorted_regrets : List[Tuple[int, float]] = [] # Each element is (scenario ID, regret) pair from a scenario
        self.max_regret_over_time : List[Tuple[int, float]] = [] # List of (scenario ID, maximum regret) attained at the given time
        self.scenario_tree_levels : List[int] = []
        self.next_temp_id = -1 # Do not access this directly, use the function
        self.failure_rate : Dict[int, int] = {} # Key = scenario ID, value = fraction of outcomes that resulted in failure

    def add_scenario(self, scenario: Scenario):
        scenario.id = len(self.scenario_regrets) # Set scenario ID here
        self.scenario_regrets[scenario.id] = scenario.avg_regret.value

        new_pair = (scenario.id, scenario.avg_regret.value)
        insert_idx = binary_search_dec_idx(self.sorted_regrets, new_pair, key = lambda x: x[1])
        self.sorted_regrets.insert(insert_idx, new_pair)

        if len(self.max_regret_over_time) == 0:
            self.max_regret_over_time.append(new_pair)
        else:
            if scenario.avg_regret.value > self.max_regret_over_time[-1][1]:
                self.max_regret_over_time.append(new_pair)
            else:
                self.max_regret_over_time.append(self.max_regret_over_time[-1])

        self.scenario_tree_levels.append(scenario.tree_level)

        nsucc, nfail = scenario.get_outcome_distribution()
        self.failure_rate[scenario.id] = nfail / (nsucc + nfail)

    def num_scenarios(self):
        return len(self.scenario_regrets)

    def get_new_temp_id(self):
        """Returns a temporary ID that can be assigned to any scenario"""
        self.next_temp_id += 1
        if self.next_temp_id < len(self.scenario_regrets):
            self.next_temp_id = len(self.scenario_regrets)
        return self.next_temp_id


class ScenarioAnalysis:
    """Class to store additional analyses about the generated scenarios."""
    def __init__(self):
        # Lists of additional information, idx = scenario ID
        self.mean_astar_replan_angle : List[float] = [] # [deg]
        self.max_astar_replan_angle : List[float] = [] # [deg]
        self.mean_path_follower_angle : List[float] = [] # [deg]
        self.max_path_follower_angle : List[float] = [] # [deg]
        self.avg_loops_per_path : List[float] = []
        self.weighted_obstacle_detection_distance : List[float] = [] # [m]

        self.mean_image_processing_durations : List[float] = []
        self.max_image_processing_durations : List[float] = []
        self.mean_astar_replan_iterations : List[float] = []
        self.max_astar_replan_iterations : List[float] = []
        self.mean_astar_replan_durations : List[float] = []
        self.max_astar_replan_durations : List[float] = []

########## CGA Classes and Functions
        
class CGAPopulationManager:
    """Class to store CGA scenario info in one place."""
    def __init__(self):
        self.population_idxs : List[List[int]] = [] # Each element is a list of scenario indeces that make up the population
        self.population_parent_idxs : List[List[Tuple[int]]] = [] # Each element is a list of tuples indicating the parents for that population
        self.avg_population_regret : List[float] = []
        self.max_population_regret : List[float] = []

    def num_populations(self):
        return len(self.population_idxs)

def get_cga_chromosome(scene_description: auto_scene_gen_msgs.SceneDescription, scenario_builder: ScenarioBuilderAndRefAgent):
    """ Get positional-only chromosomal data
    Returns:
        - chromosome
        - min_values
        - max_values
    """
    chromo = []
    min_values = []
    max_values = []
    xrange = scenario_builder.ssa_attr.x.range
    yrange = scenario_builder.ssa_attr.y.range

    for layout in scene_description.ssa_array:
        chromo += list(layout.x)
        min_values += [xrange[0]] * layout.num_instances
        max_values += [xrange[1]] * layout.num_instances

        chromo += list(layout.y)
        min_values += [yrange[0]] * layout.num_instances
        max_values += [yrange[1]] * layout.num_instances

    return np.array(chromo), np.array(min_values), np.array(max_values)

def cga_chromosome_to_scene_description(chromosome, scenario_builder: ScenarioBuilderAndRefAgent):
    scene_description = scenario_builder.create_default_scene_description_msg()
    i = 0
    for layout in scene_description.ssa_array:
        x = chromosome[i:i+layout.num_instances]
        layout.x = x.tolist()
        i += layout.num_instances

        y = chromosome[i:i+layout.num_instances]
        layout.y = y.tolist()
        i += layout.num_instances

        layout.visible = [True] * layout.num_instances
    return scene_description

def cga_mate_and_mutate(p1: np.float32, p2: np.float32, mutation_prob: float, min_values: np.float32, max_values: np.float32):
    """Create two children from mating parent1 and parent2"""
    np.random.seed()
    crossover = np.random.randint(p1.size)

    # Perform crossover
    if crossover == p1.size-1:
        c1 = p2.copy()
        c1[-1] = p1[-1]
        c2 = p1.copy()
        c2[-1] = p2[-1]
    else:
        c1 = p1.copy()
        c2 = p2.copy()
        c1[crossover+1:] = p2[crossover+1:]
        c2[crossover+1:] = p1[crossover+1:]

    # Change crossover point with linear interpolation
    beta = np.random.rand()
    c1[crossover] = (1-beta)*p1[crossover] + beta*p2[crossover]
    c2[crossover] = (1-beta)*p2[crossover] + beta*p1[crossover]

    # Mutate
    prob1 = np.random.rand(p1.size) # Mutation probability
    prob2 = np.random.rand(p1.size)
    rand1 = min_values + np.random.rand(p1.size) * (max_values - min_values) # New random values within allowed range
    rand2 = min_values + np.random.rand(p1.size) * (max_values - min_values)
    c1[prob1 < mutation_prob] = rand1[prob1 < mutation_prob]
    c2[prob2 < mutation_prob] = rand2[prob2 < mutation_prob]

    return c1, c2
    
def add_reference_trajectory_to_scenario(scenario: Scenario, scenario_request, scenario_builder: ScenarioBuilderAndRefAgent):
    """
    Args:
        - scenario: Scenario to fill in reference trajectory data
        - scenario_request: The corresponding RunScenario request
        - scenario_builder: The corresponding scenario builder
    """
    # Add opt path, time, and cost to scenario
    opt_path, opt_path_len, opt_path_cost, opt_path_cost_values = scenario_builder.get_optimal_path(scenario_request)
    scenario.opt_path = np.array(opt_path)
    scenario.opt_path_len = opt_path_len
    scenario.opt_path_cost = opt_path_cost
    scenario.opt_path_cost_values = opt_path_cost_values
    scenario.opt_path_time = opt_path_len / scenario_builder.nominal_vehicle_speed

    # Get opt time values
    opt_path_time_values = [0.]
    for i in range(len(opt_path)-1):
        edge_length_ratio = fast_norm(opt_path[i+1] - opt_path[i]) / opt_path_len
        dt = edge_length_ratio * scenario.opt_path_time
        opt_path_time_values.append(opt_path_time_values[-1] + dt)
    opt_path_time_values[-1] = scenario.opt_path_time # Explicitly set this in case there are precision errors
    scenario.opt_path_time_values = opt_path_time_values

def multiprocess_create_cga_scenarios(pid: int, parent1: Scenario, parent2: Scenario, mutation_prob: float, scenario_builder: ScenarioBuilderAndRefAgent, temp_ids: List[int], save_dir: str):
    # Assumes that only the positional (x,y) elements of the SSAs are part of the chromosome
    np.random.seed()
    
    start_location = scenario_builder.opr_attr.get_default_start_location()
    goal_location = scenario_builder.opr_attr.get_default_goal_location()

    # Create random scenarios for first population
    if parent1 is None and parent2 is None:
        for idx, temp_id in enumerate(temp_ids):
            start_time = time.time()
            
            b_feasible = False
            while not b_feasible:
                scenario_request = scenario_builder.create_default_run_scenario_request()
                for layout in scenario_request.scene_description.ssa_array:
                    for i in range(layout.num_instances):
                        # Find (x,y) coordinate outside obstacle-free radii (this is probably faster than repeatedly calling add_random_ssa())
                        while True:
                            x = scenario_builder.ssa_attr.x.random()
                            y = scenario_builder.ssa_attr.y.random()
                            if fast_norm(start_location - np.array([x,y])) <= scenario_builder.start_obstacle_free_radius:
                                continue
                            if fast_norm(goal_location - np.array([x,y])) <= scenario_builder.goal_obstacle_free_radius:
                                continue
                            break
                        layout.x[i] = x
                        layout.y[i] = y
                        layout.visible[i] = True
                b_feasible = scenario_builder.is_scenario_request_feasible(scenario_request)

            # Create scenario
            scenario = Scenario(temp_id, None, 0, scenario_request.scene_description)
            add_reference_trajectory_to_scenario(scenario, scenario_request, scenario_builder)
            file_path = os.path.join(save_dir, f"scenario_temp_{temp_id}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(scenario, f, pickle.HIGHEST_PROTOCOL)
            print(f"[Parallel Worker {pid}] ({idx+1} of {len(temp_ids)}) Child scenario w/temp ID {temp_id}) processed in {time.time()-start_time:.4f} seconds.")
        
    # Create new CGA population
    else:
        n = 0
        p1_chromo, min_chromo, max_chromo = get_cga_chromosome(parent1.scene_description, scenario_builder)
        p2_chromo, _, _ = get_cga_chromosome(parent2.scene_description, scenario_builder)
        while n < len(temp_ids):
            start_time = time.time()
            c1_chromo, c2_chromo = cga_mate_and_mutate(p1_chromo, p2_chromo, mutation_prob, min_chromo, max_chromo)

            # Chromo 1
            scenario_request = scenario_builder.create_default_run_scenario_request()
            scenario_request.scene_description = cga_chromosome_to_scene_description(c1_chromo, scenario_builder)

            if scenario_builder.is_scenario_request_feasible(scenario_request):
                scenario = Scenario(temp_ids[n], (parent1.id, parent2.id), parent1.tree_level+1, scenario_request.scene_description)
                add_reference_trajectory_to_scenario(scenario, scenario_request, scenario_builder)
                file_path = os.path.join(save_dir, f"scenario_temp_{temp_ids[n]}.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(scenario, f, pickle.HIGHEST_PROTOCOL)
                print(f"[Parallel Worker {pid}] ({n+1} of {len(temp_ids)}) Child 1 of 2 scenario w/temp ID {temp_ids[n]}) processed in {time.time()-start_time:.4f} seconds.")
                n += 1

            if n == len(temp_ids):
                break
            
            # Chromo 2
            scenario_request = scenario_builder.create_default_run_scenario_request()
            scenario_request.scene_description = cga_chromosome_to_scene_description(c2_chromo, scenario_builder)

            if scenario_builder.is_scenario_request_feasible(scenario_request):
                scenario = Scenario(temp_ids[n], (parent1.id, parent2.id), parent1.tree_level+1, scenario_request.scene_description)
                add_reference_trajectory_to_scenario(scenario, scenario_request, scenario_builder)
                file_path = os.path.join(save_dir, f"scenario_temp_{temp_ids[n]}.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(scenario, f, pickle.HIGHEST_PROTOCOL)
                print(f"[Parallel Worker {pid}] ({n+1} of {len(temp_ids)}) Child 2 of 2 scenario w/temp ID {temp_ids[n]}) processed in {time.time()-start_time:.4f} seconds.")
                n += 1

    print(f"[Parallel Worker {pid}] Done")

##########

class BACRE2DAstarTrav(AutoSceneGenClient):
    """Class that implements Black-Box Adversarially Compounding Regret Through Evolution (BACRE) and various other methods
    for testing the A* trav vehicle model on a 2D horizontal plane.
    Fully integrated with the UE4 AutomaticSceneGeneration plugin.

    NOTE: This class expects all position measurements to be in [m] and expressed in a right-handed north-west-up coordinate frame.
    """
    # Modes
    # NOTE: When using MODE_BACRE, MODE_RANDOM, or MODE_CGA the code will automatically follow with ANALYSIS, SCENE_CAPTURE, and REPLAY
    MODE_BACRE = "bacre"                        # Run BACRE algorithm to create adversarial scenarios
    MODE_RANDOM = "random"                      # Generate random scenarios
    MODE_CGA = "continuous_genetic_algorithm"   # Run CGA to create adversarial scenarios
    MODE_ANALYSIS = "analysis"                  # Analyze all generated scenarios (according to user-specific needs)
    MODE_ANALYSIS_PLOT = "analysis_plot"        # Make plots based on latest analysis
    MODE_ANALYSIS_REPLAY = "analysis_replay"    # Make plots and analyze replayed scenarios
    MODE_SCENE_CAPTURE = "scene_capture"        # Capture various point-of-views (images) from certain scenarios (see SCR_ options below)
    MODE_REPLAY = "replay"                      # Replay certain scenarios by ranking (allows you to save/log additional vehicle node data for further analysis)

    # Parent selection options
    PAR_SELECT_UPPER_QUANTILE = "upper_quantile_search"
    PAR_SELECT_RCR = "regret_curation_ratio"
    PAR_SELECT_METHODS = (PAR_SELECT_UPPER_QUANTILE, PAR_SELECT_RCR)

    # Scene capture request (SCR) options
    SCR_ID = "id"   # Capture scenes by ID
    SCR_BEST = "best" # Capture scenes based on the best X scenarios (i.e, with the highest X regret values)
    SCR_BEST_FAMILY_TREE = "best_family_tree" # Capture scenes for the family trees of the best X scenarios

    # Variables used to make the basic scene plot (not the pretty one)
    SSA_TYPES = ('tree', 'bush', 'rock')
    SSA_PLOT_INFO = {
        'tree': {"color": "forestgreen", "marker": "^", "markersize": 8**2}, # Forest green, tri_up
        'bush': {"color": "lime", "marker": "p", "markersize": 8**2}, # Brown RGB(165,42,42), pentagon
        'rock': {"color": (0.5, 0.5, 0.5, 1.), "marker": "o", "markersize": 8**2},  # Gray, circle
    }

    # Colors used in plotting
    COLOR_OBS_FREE_ZONE = "black"
    COLOR_START_LOCATION = "yellow"
    COLOR_GOAL_LOCATION = "blue"
    COLOR_OPT_REF_PATH = "black"
    COLOR_BASE_AV_PATH = "cyan"
    COLOR_TEST_AV_PATH = "red"
    COLOR_LEGEND ="white"

    PLOT_PADDING = 5.
    MAX_RERUN_REQUESTS = 3 # Maximum number of times the autonomy system can request a rerun for the last scenario. Can sometimes be useful if there actually was a problem (e.g., due to ROS2)

    def __init__(self, 
                auto_scene_gen_client_dict: Dict,               # Dictionary of parameters for the base AutoSceneGenClient
                mode: str,                                      # Primary mode of operation, see MODE_* options above
                num_runs_per_scenario: int,                     # Number of times we run each scenario for to predict the difficulty for the test vehicle
                num_base_env: int,                              # Number of base or "seed" environments to create
                num_env_to_generate: int,                       # Maximum number of environments to generate
                num_mutations: int,                             # Number of mutations made when creating a child scenario
                b_regret_uses_equivalent_x_errors: bool,        # Indicates if the regret calculation should ignore the equivalent-x errors
                initial_prob_editing_random_env: float,         # Initial probability of choosing a random parent environment to edit
                prob_editing_random_env: float,                 # Probability of choosing a random parent environment to edit (over all generated levels)
                upper_search_quantile: float,                   # Specifies the upper quantile of high-regret scenarios from which to randomly choose from
                initial_regret_curation_ratio: float,           # Specifies the initial upper fraction of the population's regret range from which to curate scenarios from
                regret_curation_ratio: float,                   # Specifies the upper fraction of the population's regret range from which to curate scenarios from
                anneal_duration: int,                           # Number of iterations to anneal the above hyperparameters over (after creating the base population). Applies to 'prob_editing_random_env' and 'regret_curation_ratio'.
                parent_selection_method: str,                   # Method for selecting a scenario's parent
                ssa_add_prob: float,                            # Nominal probability for adding an SSA
                ssa_remove_prob: float,                         # Nominal probability for perturbing an existing SSA
                ssa_perturb_prob: float,                        # Nominal probability for removing an SSA
                data_save_freq: int,                            # Save data manager at this frequency of generated scenarios
                num_difficult_scenarios_to_plot: int,           # Number of most difficult scenarios to plot
                # CGA params
                cga_population_size: int,                       # Size of each population for the CGA method
                cga_num_families_per_population: int,           # Number of families to use for creating new populations from. The top 2f scenarios are parents which are paired like so: (1,2), (3,4), (5,6), ...
                cga_num_populations: int,                       # Number of populations to run CGA method for
                cga_mutation_prob: float,                       # Probability that an individual attribute in the scenario desription is mutated to a new random value
                # Scene capture
                scene_capture_settings: auto_scene_gen_msgs.SceneCaptureSettings,   # The scene capture settings message to use
                scene_capture_request: str,                     # The scene capture request, see SCR_* options above
                scene_capture_request_value,                    # The value assoicated with the scene capture request
                # Replay request
                replay_request_rank_range: List,                # A list of two items, in non-decreasing order, indicating the range of the best scenarios to replay
                replay_request_num_replays: int,                # The number of times to replay each scenario
                ):
        super().__init__(**auto_scene_gen_client_dict)
        np.random.seed() # Pick new random seed each time we load

        # BACRE params
        self.num_runs_per_scenario = num_runs_per_scenario
        self.num_base_env = num_base_env
        self.num_env_to_generate = num_env_to_generate
        self.num_mutations = num_mutations
        self.b_regret_uses_equivalent_x_errors = b_regret_uses_equivalent_x_errors
        self.initial_prob_editing_random_env = initial_prob_editing_random_env
        self.prob_editing_random_env = prob_editing_random_env
        self.upper_search_quantile = upper_search_quantile
        self.initial_regret_curation_ratio = initial_regret_curation_ratio
        self.regret_curation_ratio = regret_curation_ratio
        self.anneal_duration = anneal_duration

        if parent_selection_method not in self.PAR_SELECT_METHODS:
            raise ValueError(f"'parent_selection_method' must be one of {self.PAR_SELECT_METHODS}.")
        self.parent_selection_method = parent_selection_method

        if math.fabs(ssa_add_prob + ssa_remove_prob + ssa_perturb_prob - 1.) > 0.001:
            raise ValueError(f"SSA add/remove/perturb probabilities must sum to 1.")
        self.ssa_add_prob = ssa_add_prob
        self.ssa_remove_prob = ssa_remove_prob
        self.ssa_perturb_prob = ssa_perturb_prob
        
        self.data_save_freq = data_save_freq
        self.num_difficult_scenarios_to_plot = num_difficult_scenarios_to_plot

        # Create main directories
        self.dirs = {}
        self.dirs["data"] = os.path.join(self.main_dir, "data")
        self.dirs["scenarios"] = os.path.join(self.main_dir, "data", "scenarios")
        self.dirs["scene_captures"] = os.path.join(self.main_dir, "data", "scene_captures")
        self.dirs["figures"] = os.path.join(self.main_dir, "figures")
        self.dirs["legends"] = os.path.join(self.main_dir, "figures", "legends")
        self.dirs["replay"] = os.path.join(self.main_dir, "replay")
        self.dirs["outer"] = os.path.abspath(os.path.join(self.main_dir, "..")) # Contains all experiments
        self.dirs["group_summary"] = os.path.join(self.dirs["outer"], "summary")
        for _,dir in self.dirs.items():
            os.makedirs(dir, exist_ok = True)

        # Create temporary vehicle node save directories
        self.temp_vehicle_node_save_dirs = {}
        for wid in self.recognized_wids:
            self.temp_vehicle_node_save_dirs[wid] = os.path.join(self.dirs["data"], f"vehicle_node_worker{wid}")
        for _,dir in self.temp_vehicle_node_save_dirs.items():
            os.makedirs(dir, exist_ok = True)

        self.max_regret_achieved = -math.inf

        # Mode-specific processing ID (mspid) variables
        self.worker_mspids : Dict[int, int] = {} # Key = worker ID, Value = an mspid
        self.mspids : List[int] = [] # List of mspids that need to be run (gets set once)
        self.mspids_completed : List[int] = [] # List of all completed mspids (new mspids get added to this)

        # CGA variables
        # NOTE: mspids are the scenario temp IDs to run for the current CGA population
        self.cga_population_size = cga_population_size
        self.cga_num_families_per_population = cga_num_families_per_population
        self.cga_num_populations = cga_num_populations
        self.cga_mutation_prob = cga_mutation_prob
        self.cga_temp_save_dir = os.path.join(self.main_dir, "data", "cga_temp")
        os.makedirs(self.cga_temp_save_dir, exist_ok = True)
        self.cga_population_manager_filepath = os.path.join(self.main_dir, "data", "cga_population_manager.pkl")
        self.cga_population_manager, b_loaded = self.load_cga_population_manager(self.cga_population_manager_filepath)
        if b_loaded:
            self.log("info", f"Loaded CGA population manager from: {self.cga_population_manager_filepath}")
        else:
            self.log("info", f"Could not load CGA population manager. Creating one from scratch.")
        self.cga_population_idxs = [] # Scenario IDs for the current population
        self.cga_parent_ids = [] # List of tuples denoting the parent combinations for this population (no need to rerun these scenarios)

        # Scene capture mode variables
        # NOTE: mspids are the scenario IDs to use for scene capture
        self.scene_capture_settings = scene_capture_settings
        self.scene_capture_request = scene_capture_request
        self.scene_capture_request_value = scene_capture_request_value

        # Replay mode variables
        # NOTE: mspids are the scenario indeces within the self.replay_scenarios variable
        self.replay_request_rank_range = replay_request_rank_range
        self.replay_request_num_replays = replay_request_num_replays
        self.replay_scenarios : List[Scenario] = []

        self.b_saved_scene_legends = False
        self.b_done_testing = False

        self.scenario_regret_manager_filepath = os.path.join(self.main_dir, "data", "scenario_regret_manager.pkl")
        self.scenario_regret_manager, b_loaded = self.load_scenario_regret_manager(self.scenario_regret_manager_filepath)
        self.scenario_analysis_filepath = os.path.join(self.main_dir, "data", "scenario_analysis.pkl")
        self.scenario_analysis = None
        self.generated_scenarios : Dict[int, Scenario] = {} # Will use deferred loading for specific scenarios as needed
        if b_loaded:
            self.log("info", f"Loaded scenario regret manager from: {self.scenario_regret_manager_filepath}")
            self.max_regret_achieved = self.scenario_regret_manager.sorted_regrets[0][1]

            self.scenario_analysis, b_loaded2 = self.load_scenario_analysis(self.scenario_analysis_filepath)
            if b_loaded2:
                self.log("info", f"Loaded scenario analysis")
            else:
                self.log("info", f"Could not load scenario analysis")
        else:
            self.log("info", f"Could not load scenario regret manager from file. Creating one from scratch.")

        # Mode of operation
        self.mode = {
            self.MODE_BACRE: False,
            self.MODE_RANDOM: False,
            self.MODE_CGA: False,
            self.MODE_ANALYSIS: False,
            self.MODE_ANALYSIS_PLOT: False,
            self.MODE_ANALYSIS_REPLAY: False,
            self.MODE_SCENE_CAPTURE: False,
            self.MODE_REPLAY: False,
        }
        self.b_in_testing_mode = False

        mode = str.lower(mode)
        if mode not in self.mode.keys():
            raise ValueError(f"Unrecognized mode of operation '{mode}'")

        # Uncomment when needed, can take a few minutes to run
        # self.plot_max_regret_scenario_tree()
        # self.plot_max_regret_scenes()
        # self.plot_scenario_rankings()

        self.log("info", f"Configuringing mode: {mode}")
        if mode == self.MODE_BACRE:
            self.mode[self.MODE_BACRE] = True
            self.b_in_testing_mode = True
            if self.scenario_regret_manager.num_scenarios() >= self.num_env_to_generate:
                self.log("info", f"Done creating scenarios.")
                self.mode[self.MODE_BACRE] = False
                self.b_in_testing_mode = False
            self.plot_scenario_rankings()
            self.plot_max_regret_scene()
        
        elif mode == self.MODE_RANDOM:
            self.mode[self.MODE_RANDOM] = True
            self.b_in_testing_mode = True
            if self.scenario_regret_manager.num_scenarios() >= self.num_env_to_generate:
                self.log("info", f"Done creating scenarios.")
                self.mode[self.MODE_RANDOM] = False
                self.b_in_testing_mode = False
            self.plot_scenario_rankings()
            self.plot_max_regret_scene()

        elif mode == self.MODE_CGA:
            self.mode[self.MODE_CGA] = True
            self.b_in_testing_mode = True
            
            self.log("info", f"Created {self.cga_population_manager.num_populations()} populations")
            if self.cga_population_manager.num_populations() >= self.cga_num_populations:
                self.mode[self.MODE_CGA] = False
                self.b_in_testing_mode = False
                self.log("info", f"Done creating CGA populations.")

            self.plot_scenario_rankings()
            self.plot_max_regret_scene()
                
        elif mode == self.MODE_ANALYSIS:
            self.generated_scenarios : Dict[int, Scenario] = self.load_generated_scenarios(self.dirs["scenarios"], self.scenario_regret_manager)
            self.plot_scenario_rankings()
            self.plot_max_regret_scene()
            self.analyze_generated_scenarios_snapshots()

        elif mode == self.MODE_ANALYSIS_PLOT:
            self.plot_scenario_rankings()
            self.plot_max_regret_scene()
            self.plot_scenario_analysis()

        elif mode == self.MODE_ANALYSIS_REPLAY:
            self.reanalyze_replay_scenarios()

        elif mode == self.MODE_SCENE_CAPTURE:
            self.mode[self.MODE_SCENE_CAPTURE] = True
            self.process_scene_capture_request()

        elif mode == self.MODE_REPLAY:
            self.mode[self.MODE_REPLAY] = True
            self.process_replay_request()

        else:
            raise ValueError(f"Unrecognized mode of operation '{mode}'")

        self.log("info", "Initialized BACRE 2D")
        self.log("info", "-" * 60, b_log_ros=False)

    def num_mspids_remaining(self):
        """Return the number of mode-specific pids that have not been processed yet"""
        return len(self.mspids) - len(self.mspids_completed) - len(self.worker_mspids)

    def get_next_mspid(self):
        """Return the next mode-specific pid for use"""
        if self.num_mspids_remaining() == 0:
            return None
        
        for id in self.mspids:
            if id not in self.mspids_completed and id not in list(self.worker_mspids.values()):
                return id

    def load_scenario_regret_manager(self, file_path: str):
        """Load scenario regret manager from file, if it exists.
        
        Args:
            - file_path: File to laod data manager from

        Returns:
            - Scenario regret manager
            - Bool indicating if load was successful
        """
        manager = ScenarioRegretManager()
        b_loaded = False
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                manager : ScenarioRegretManager = pickle.load(f)
            b_loaded = True
        return manager, b_loaded

    def save_scenario_regret_manager(self):
        """Save the internal scenario regret manager"""
        with open(self.scenario_regret_manager_filepath, 'wb') as f:
            pickle.dump(self.scenario_regret_manager, f, pickle.HIGHEST_PROTOCOL)
        self.log("info", f"Saved scenario regret manager to file: {self.scenario_regret_manager_filepath}")

    def load_cga_population_manager(self, file_path: str):
        """Load CGA population manager from file, if it exists.
        
        Args:
            - file_path: File to laod data manager from

        Returns:
            - CGA population manager
            - Bool indicating if load was successful
        """
        manager = CGAPopulationManager()
        b_loaded = False
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                manager : CGAPopulationManager = pickle.load(f)
            b_loaded = True
        return manager, b_loaded

    def save_cga_population_manager(self):
        """Save the internal CGA population manager"""
        with open(self.cga_population_manager_filepath, 'wb') as f:
            pickle.dump(self.cga_population_manager, f, pickle.HIGHEST_PROTOCOL)
        self.log("info", f"Saved CGA population manager to file: {self.cga_population_manager_filepath}")

    def load_cga_temp_scenario(self, temp_id: int):
        """Load a scenario from teh CGA temp directory with the given temp ID"""
        scenario = None
        file_path = os.path.join(self.cga_temp_save_dir, f"scenario_temp_{temp_id}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                scenario : Scenario = pickle.load(f)
        return scenario

    def load_scenario_analysis(self, file_path: str):
        """Load scenario analysis from file, if it exists.
        
        Args:
            - file_path: File to laod data manager from

        Returns:
            - The scenario analysis
            - Bool indicating if load was successful
        """
        analysis = ScenarioAnalysis()
        b_loaded = False
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                analysis : ScenarioAnalysis = pickle.load(f)
            b_loaded = True
        return analysis, b_loaded

    def save_generated_scenario(self, scenario: Scenario):
        """Save a scenario that was generated during the search process.
        
        Args:
            - scenario: The scenario to save in the data/scenarios folder
        """
        if scenario.id is None:
            self.log("error", f"Cannot save scenario with ID None.")
            return
        
        file_path = os.path.join(self.dirs["scenarios"], f"scenario_{scenario.id}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(scenario, f, pickle.HIGHEST_PROTOCOL)
        self.log("info", f"Saved scenario with ID {scenario.id} to file: {file_path}")

    def load_generated_scenarios(self, scenario_dir: str, scenario_regret_manager: ScenarioRegretManager):
        """Load all data scenarios from the given directory. 
        If a file does not exist, the code will raise an error. At that point you may need to sstart over.
        
        Args:
            - scenario_dir: Directory to find all the scenarios
            - scenario_regret_manager: The scenario regret manager that accompanies the scenarios

        Returns:
            A dictionary of the scenarios with the keys being the scenario IDs
        """
        if scenario_regret_manager.num_scenarios() == 0:
            self.log("info", f"No generated scenarios to load.")
            return {}

        start_time = time.time()
        self.log("info", f"Loading generated scenarios...")
        scenario_dict : Dict[int, Scenario] = {}

        for id in range(scenario_regret_manager.num_scenarios()):
            file_path = os.path.join(scenario_dir, f"scenario_{id}.pkl")
            with open(file_path, 'rb') as f:
                scenario_dict[id] = pickle.load(f)

        self.log("info", f"Loaded {scenario_regret_manager.num_scenarios()} scenarios in {time.time() - start_time:.4f} seconds.")
        return scenario_dict
    
    def get_generated_scenario_from_id(self, id: int):
        """Load and return a generated scenario using the parameters associated with this class. For internal use only.
        
        Args:
            id: The ID for the desired scenario to load and return

        Returns:
            The specified scenario
        """
        self.load_generated_scenario_from_id(id, self.dirs["scenarios"], self.scenario_regret_manager, self.generated_scenarios)
        return self.generated_scenarios[id]
        
    def load_generated_scenario_from_id(self, id: int, scenario_dir: str, scenario_regret_manager: ScenarioRegretManager, generated_scenarios: Dict[int, Scenario]):
        """Generic function to load a generated scenario from an ID
        
        Args:
            - id: Scenario ID to load
            - scenario_dir: Directory to load the scenario from
            - scenario_regret_manager: Accompanying scenario regret manager
            - generated_scenarios: Dictionary to add the scenario to, if it exists
        """
        if 0 <= id < scenario_regret_manager.num_scenarios():
            if id not in generated_scenarios.keys():
                file_path = os.path.join(scenario_dir, f"scenario_{id}.pkl")
                with open(file_path, 'rb') as f:
                    generated_scenarios[id] = pickle.load(f)
    
    def process_replay_request(self):
        """Process the replay request and setup the replay scenarios.

        Possible requests:
            - rankings: Specify a range of rankings to replay
            - scenario: Specify a specific scenario ID to replay
        
        If you want to replay scenarios from a different experiment (with the same parameters), add a 'dir' key with the relative path to that desired experiment directory.
        Do not add the 'data' substring to that path as this function will take care of that. 
        """
        self.replay_scenarios.clear()
        self.num_replays_per_scenario = self.replay_request_num_replays

        self.worker_mspids.clear()
        self.mspids.clear()
        self.mspids_completed.clear()

        scenario_regret_manager = self.scenario_regret_manager
        scenario_dir = self.dirs["scenarios"]

        if scenario_regret_manager.num_scenarios() == 0:
            self.log("warn", f"Cannot use replay mode because there are no scenarios to replay.")
            self.mode[self.MODE_REPLAY] = False
            return

        # Replay rank range
        self.log("info", f"Processing replay request for rank range")
        rank_range = self.replay_request_rank_range

        if rank_range[0] > rank_range [1]:
            raise ValueError(f"Replay Request: Invalid range of rankings {rank_range}. {rank_range[0]} is not <= {rank_range[1]}.")
        
        if rank_range[0] < 1 or rank_range[1] > scenario_regret_manager.num_scenarios():
            raise ValueError(f"Replay Request: Invalid range of rankings {rank_range}. Valid numbers are integers from 1 to {scenario_regret_manager.num_scenarios()}.")
                    
        generated_scenarios : Dict[int, Scenario] = {}
        replay_pairs = scenario_regret_manager.sorted_regrets[rank_range[0]-1:rank_range[1]]
        try:
            for id, regret in replay_pairs:
                self.load_generated_scenario_from_id(id, scenario_dir, scenario_regret_manager, generated_scenarios)
                self.replay_scenarios.append(generated_scenarios[id])
        except Exception:
            self.log("warn", f"Exception when loading scenario IDs: {traceback.format_exc()}")
            self.mode[self.MODE_REPLAY] = False
            return
        self.replay_rankings = list(range(rank_range[0], rank_range[1]+1))

        self.mspids = np.arange(len(self.replay_scenarios)).tolist() # Scenario indeces within the self.replay_scenarios variable
        self.log("info", f"Number of scenarios to replay: {len(self.mspids)}")

    def process_scene_capture_request(self):
        """Process the scene capture request so we can capture the desired scene images"""
        self.log("info", f"Processing scene capture request: {self.scene_capture_request}")
        val = self.scene_capture_request_value
        scene_capture_scenario_ids = []

        self.worker_mspids.clear()
        self.mspids.clear()
        self.mspids_completed.clear()

        if self.scene_capture_request == self.SCR_ID:
            if isinstance(val, int):
                if 0 <= val < self.scenario_regret_manager.num_scenarios():
                    scene_capture_scenario_ids = [val]
                else:
                    raise ValueError(f"Scene capture request '{self.SCR_ID}': Scenario ID {val} does not exist. Choose from 0 to {self.scenario_regret_manager.num_scenarios()-1}.")
            
            elif isinstance(val, tuple) or isinstance(val, list):
                if len(val) != 2:
                    raise ValueError(f"Scene capture request '{self.SCR_ID}': tuple/list must have two nonnegative integers in it.")
                if 0 <= val[0] <= val[1] and val[1] < self.scenario_regret_manager.num_scenarios():
                    scene_capture_scenario_ids = list(range(val[0], val[1]+1))
                else:
                    raise ValueError(f"Scene capture request '{self.SCR_ID}': Tuple/list must be of the form (start_id, stop_id), with 0 <= start_id <= stop_id < {self.scenario_regret_manager.num_scenarios()}.")
            
            elif isinstance(val, str):
                if val.lower() == "all":
                    scene_capture_scenario_ids = list(range(0, self.scenario_regret_manager.num_scenarios()))
                else:
                    raise ValueError(f"Scene capture request '{self.SCR_ID}': Unrecognized string value '{val}'")

            else:
                raise ValueError(f"Scene capture request '{self.SCR_ID}': Unrecognized value {val}")
        
        elif self.scene_capture_request == self.SCR_BEST:
            if isinstance(val, int):
                if 0 < val <= self.scenario_regret_manager.num_scenarios():
                    scene_capture_scenario_ids = []
                    for i in range(val):
                        id, _ = self.scenario_regret_manager.sorted_regrets[i]
                        scene_capture_scenario_ids.append(id)
                else:
                    raise ValueError(f"Scene capture request '{self.SCR_BEST}': Integer value must be in the range [1, {self.scenario_regret_manager.num_scenarios()}].")
            else:
                raise ValueError(f"Scene capture request '{self.SCR_BEST}': Only accept an integer in the range [1, {self.scenario_regret_manager.num_scenarios()}].")
        
        elif self.scene_capture_request == self.SCR_BEST_FAMILY_TREE:
            if isinstance(val, int):
                if 0 < val <= self.scenario_regret_manager.num_scenarios():
                    scene_capture_scenario_ids = []
                    for i in range(val):
                        id, _ = self.scenario_regret_manager.sorted_regrets[i]
                        if id not in scene_capture_scenario_ids:
                            scene_capture_scenario_ids.append(id)

                        parent_id = self.get_generated_scenario_from_id(id).parent_id
                        while parent_id is not None and isinstance(parent_id, int): # TODO: Account for parent ID being a tuple (for CGA)
                            if parent_id not in scene_capture_scenario_ids:
                                scene_capture_scenario_ids.append(parent_id)
                            parent = self.get_generated_scenario_from_id(parent_id)
                            parent_id = parent.parent_id
                else:
                    raise ValueError(f"Scene capture request '{self.SCR_BEST_FAMILY_TREE}': Integer value must be in the range [1, {self.scenario_regret_manager.num_scenarios()}].")
            else:
                raise ValueError(f"Scene capture request '{self.SCR_BEST_FAMILY_TREE}': Only accept an integer in the range [1, {self.scenario_regret_manager.num_scenarios()}].")
        
        else:
            raise ValueError(f"Scene capture request: Unrecognized request type '{self.scene_capture_request}'.")
        
        self.mspids = scene_capture_scenario_ids # The actual scenario IDs for scene capture
        self.log("info", f"Number of scenarios to recreate for scene capture: {len(self.mspids)}")

    def plot_scenario_rankings(self):
        """Plot the regret of all scenarios in decreasing order"""
        self.log("info", "Plotting scenario rankings...")
        if self.scenario_regret_manager.num_scenarios() == 0:
            self.log("info", "No ranked scenarios to plot from. Skipping task.")
            return
        
        np_sorted_regrets = np.array(self.scenario_regret_manager.sorted_regrets)
        regrets = np_sorted_regrets[:,1]

        np_max_regret_over_time = np.array(self.scenario_regret_manager.max_regret_over_time)
        max_regret_over_time = np_max_regret_over_time[:,1]

        mvg_avg_window = 100
        regret_over_time = np.array(list(self.scenario_regret_manager.scenario_regrets.values()))
        smoothed_regret_over_time = get_smoothing_avg(regret_over_time, mvg_avg_window)

        # Sort tree levels correctly
        sorted_tree_levels = []
        for id,regret in self.scenario_regret_manager.sorted_regrets:
            sorted_tree_levels.append(self.scenario_regret_manager.scenario_tree_levels[id])

        # Plot regret rankings
        fontsize = 12
        legend_fontsize = 12
        rankings = np.arange(self.scenario_regret_manager.num_scenarios()) + 1 # list(range(1, self.data_manager.num_scenarios()+1))
        fig, ax = plt.subplots(num=1)
        ax.plot(rankings, regrets, linewidth=1, color='black')
        ax.set_xlabel("Scenario Ranking Index", fontsize=fontsize)
        ax.set_ylabel("Regret", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        fig.tight_layout()
        fig.savefig(os.path.join(self.dirs['figures'], 'scenario_rankings.pdf'))
        fig.savefig(os.path.join(self.dirs['figures'], 'scenario_rankings.png'))
        fig.clf()

        # Plot regret rankings and tree levels
        if not self.mode[self.MODE_RANDOM]:
            fig, ax = plt.subplots(2,1,num=1)
            # 1. Regret
            ax[0].plot(rankings, regrets, linewidth=1, color='black')
            ax[0].set_xlabel("Scenario Ranking Index", fontsize=fontsize)
            ax[0].set_ylabel("Regret", fontsize=fontsize)
            ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
            # 2. Scenario Tree Level
            ax[1].scatter(rankings, sorted_tree_levels, s=1, color="Black")
            ax[1].set_xlabel("Scenario Ranking Index", fontsize=fontsize)
            ax[1].set_ylabel("Tree Level", fontsize=fontsize)
            ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
            fig.tight_layout()
            fig.savefig(os.path.join(self.dirs['figures'], 'scenario_rankings_and_tree_levels.pdf'))
            fig.savefig(os.path.join(self.dirs['figures'], 'scenario_rankings_and_tree_levels.png'))
            fig.clf()
        
        # Plot regret vs. ranking for first X best scenarios
        for num_best in (500,100):
            if self.mode[self.MODE_RANDOM]:
                fig, ax = plt.subplots(num=1)
                # 1. Regret
                ax.plot(rankings[:num_best], regrets[:num_best], linewidth=1, color='black')
                ax.set_xlabel("Scenario Ranking Index", fontsize=fontsize)
                ax.set_ylabel("Regret", fontsize=fontsize)
                ax.tick_params(axis="both", which="major", labelsize=fontsize)
            else:
                fig, ax = plt.subplots(2,1,num=1)
                # 1. Regret
                ax[0].plot(rankings[:num_best], regrets[:num_best], linewidth=1, color='black')
                ax[0].set_xlabel("Scenario Ranking Index", fontsize=fontsize)
                ax[0].set_ylabel("Regret", fontsize=fontsize)
                ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
                # 2. Scenario Tree Level
                ax[1].scatter(rankings[:num_best], sorted_tree_levels[:num_best], s=1, color="Black")
                ax[1].set_xlabel("Scenario Ranking Index", fontsize=fontsize)
                ax[1].set_ylabel("Tree Level", fontsize=fontsize)
                ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
            fig.tight_layout()
            fig.savefig(os.path.join(self.dirs['figures'], f'scenario_rankings_top{num_best}.pdf'))
            fig.savefig(os.path.join(self.dirs['figures'], f'scenario_rankings_top{num_best}.png'))
            fig.clf()

        # Plot histogram of regrets
        fig, ax = plt.subplots(num=1)
        ax.hist(regrets, bins=50)
        ax.set_xlabel("Regret", fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        fig.tight_layout()
        fig.savefig(os.path.join(self.dirs['figures'], 'scenario_regret_histogram.pdf'))
        fig.savefig(os.path.join(self.dirs['figures'], 'scenario_regret_histogram.png'))
        fig.clf()

        # Plot regret vs number of scenarios
        # fontsize2 = 16
        # legend_fontsize2 = 16
        fig, ax = plt.subplots(num=10)
        ax.axvline(self.num_base_env, linewidth=1, color="red", label="Num. Base Scenarios")
        if self.cga_population_manager.num_populations() == 0:
            ax.plot(rankings, smoothed_regret_over_time, linewidth=1, color="blue", label=f"Regret") # f"Regret (Mvg. Avg. Window L={mvg_avg_window})"
        ax.plot(rankings, max_regret_over_time, linewidth=1, color="black", label="Max Regret")
        ax.set_xlabel(f"Iterations", fontsize=fontsize)
        ax.set_ylabel("Regret", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        # legend = ax.legend(loc='lower center', fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.025)) # Place legend outside plot
        fig.tight_layout()
        fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_iterations.pdf'))
        fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_iterations.png'))
        ax.legend(fontsize=legend_fontsize) # "lower right"
        fig.tight_layout()
        fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_iterations_wlegend.pdf'))
        fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_iterations_wlegend.png'))

        ax.set_visible(False)
        fig.legend(fontsize=legend_fontsize)
        fig.savefig(os.path.join(self.dirs["legends"], 'regret_vs_iterations_legend.pdf'), bbox_inches="tight")
        fig.savefig(os.path.join(self.dirs["legends"], 'regret_vs_iterations_legend.png'), bbox_inches="tight")
        fig.clf()

        # Plot regret vs. CGA populations
        if self.cga_population_manager.num_populations() > 0:
            fig, ax = plt.subplots(num=1)
            pop_range = np.arange(self.cga_population_manager.num_populations()) + 1
            ax.plot(pop_range, self.cga_population_manager.max_population_regret, linewidth=1, color="black", label="Max Regret")
            ax.plot(pop_range, self.cga_population_manager.avg_population_regret, linewidth=1, color="blue", label="Avg. Regret")
            ax.set_xlabel(f"Population", fontsize=fontsize)
            ax.set_ylabel("Regret", fontsize=fontsize)
            ax.tick_params(axis="both", which="major", labelsize=fontsize)

            fig.tight_layout()
            fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_cga_populations.pdf'), bbox_inches="tight")
            fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_cga_populations.png'), bbox_inches="tight")

            ax.legend(fontsize=legend_fontsize) # "lower right"
            fig.tight_layout()
            fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_cga_populations_wlegend.pdf'), bbox_inches="tight")
            fig.savefig(os.path.join(self.dirs['figures'], 'regret_vs_cga_populations_wlegend.png'), bbox_inches="tight")

            ax.set_visible(False)
            fig.legend(fontsize=legend_fontsize)
            fig.savefig(os.path.join(self.dirs["legends"], 'regret_vs_cga_populations_legend.pdf'), bbox_inches="tight")
            fig.savefig(os.path.join(self.dirs["legends"], 'regret_vs_cga_populations_legend.png'), bbox_inches="tight")
            fig.clf()

        if self.scenario_regret_manager.num_scenarios() >= 100 and self.cga_population_manager.num_populations() == 0:
            # Plot curves showing the regret over the perturbations. Plot all the scenes from base scenario to final child.
            for num_best in (100,50,20):
                fig, ax = plt.subplots(num=1)

                for i in range(num_best):
                    id, _ = self.scenario_regret_manager.sorted_regrets[i]
                    scenario = self.get_generated_scenario_from_id(id)
                    regrets_over_time = [scenario.avg_regret.value]
                    parent_id = scenario.parent_id
                    while parent_id is not None:
                        parent = self.get_generated_scenario_from_id(parent_id)
                        regrets_over_time.insert(0, parent.avg_regret.value)
                        parent_id = parent.parent_id

                    ax.plot(np.arange(len(regrets_over_time)) + 1, regrets_over_time, linewidth=1)
                
                ax.set_xlabel(f"Tree Level", fontsize=fontsize)
                ax.set_ylabel(f"Regret", fontsize=fontsize)
                ax.tick_params(axis="both", which="major", labelsize=fontsize)
                fig.tight_layout()
                fig.savefig(os.path.join(self.dirs['figures'], f"regret_vs_tree_level_top{num_best}.pdf"))
                fig.savefig(os.path.join(self.dirs['figures'], f"regret_vs_tree_level_top{num_best}.png"))
                fig.clf()

        self.log("info", "Plotted scenario rankings")

    def plot_max_regret_scene(self):
        if self.scenario_regret_manager.num_scenarios() == 0:
            self.log("info", "No scenario data to plot from. Cannot plot max regret scene.")
            return
        
        best_scenario_id, _ = self.scenario_regret_manager.sorted_regrets[0]
        best_scenario = self.get_generated_scenario_from_id(best_scenario_id)
        self.max_regret_achieved = best_scenario.avg_regret.value
        self.log("info", f"Max regret achieved: {self.max_regret_achieved:.4f}")

        file_name = os.path.join(self.dirs['figures'], 'scene_max_regret')
        self.plot_scene_from_scenario(best_scenario, file_name)

        info_file_name = os.path.join(self.dirs['figures'], 'scene_max_regret_info.txt')
        with open(info_file_name, "w") as f:
            f.write(f"ID {best_scenario_id} / Max Regret: {self.max_regret_achieved:.4f}")

    def plot_max_regret_scenario_tree(self):
        self.log("info", f"Plotting scenario tree for max regret scenario...")
        best_scenario_id, _ = self.scenario_regret_manager.sorted_regrets[0]
        best_scenario = self.get_generated_scenario_from_id(best_scenario_id)
        self.max_regret_achieved = best_scenario.avg_regret.value
        self.plot_scenario_tree(best_scenario, "max_regret_scenario")

    def plot_max_regret_scenes(self):
        """Plot some of the highest regret scenarios"""
        self.log("info", f"Plotting max regret scenarios...")
        plot_dir = os.path.join(self.dirs['figures'], 'max_regret_scenarios')

        # Delete existing directory, if it exists, then remake it
        # if os.path.isdir(plot_dir):
        #     shutil.rmtree(plot_dir)
        os.makedirs(plot_dir, exist_ok = True)

        if self.scenario_regret_manager.num_scenarios() == 0:
            self.log("info", "No ranked scenarios to plot from. Skipping task.")
            return

        # Plot max regret scenario
        best_scenario_id, _ = self.scenario_regret_manager.sorted_regrets[0]
        best_scenario = self.get_generated_scenario_from_id(best_scenario_id)
        self.max_regret_achieved = best_scenario.avg_regret.value
        self.log("info", f"Max regret achieved: {self.max_regret_achieved:.4f}")
        
        file_name = os.path.join(self.dirs['figures'], 'scene_max_regret')
        self.plot_scene_from_scenario(best_scenario, file_name)

        # file_name = os.path.join(self.dirs['figures'], 'max_regret_scenario_tree.pdf')
        # self.plot_scenario_tree(best_scenario, file_name)

        # Plot top X ranked scenarios
        num_scenarios_to_plot = min(self.num_difficult_scenarios_to_plot, self.scenario_regret_manager.num_scenarios())
        for i in range(num_scenarios_to_plot):
            id, regret = self.scenario_regret_manager.sorted_regrets[i]
            scenario = self.get_generated_scenario_from_id(id)
            scene_plot_dir = os.path.join(plot_dir, f"rank_{i+1}")
            os.makedirs(scene_plot_dir, exist_ok = True)

            self.log("info", f"Making plots for scene rank {i+1}...")
            file_name = os.path.join(scene_plot_dir, f'scene_rank_{i+1}_id_{id}')
            self.plot_scene_from_scenario(scenario, file_name)

            info_file_name = os.path.join(scene_plot_dir, f'regret_info.txt')
            with open(info_file_name, "w") as f:
                f.write(f"Rank {i+1} / ID {id} / Regret: {regret:.4f}")

            # Plot curves showing the regret over the perturbations. Plot all the scenes from base scenario to final child.
            if self.cga_population_manager.num_populations() == 0:
                regrets_over_time = [scenario.avg_regret.value]
                parent_id = scenario.parent_id
                while parent_id is not None:
                    parent = self.get_generated_scenario_from_id(parent_id)
                    regrets_over_time.insert(0, parent.avg_regret.value)
                    parent_id = parent.parent_id
                
                fig, ax = plt.subplots(num=2)
                ax.plot(np.arange(len(regrets_over_time)) + 1, regrets_over_time, linewidth=1, color="black")
                ax.set_xlabel(f"Tree Level")
                ax.set_ylabel(f"Regret")
                fig.tight_layout()
                plt.savefig(os.path.join(scene_plot_dir, f"scene_rank_{i+1:02d}_regret_vs_tree_level.pdf"))
                plt.clf()

    def plot_scenario_tree(self, scenario: Scenario, scenario_name: str):
        """Plot the entire set of scenes from the scenario tree, from base scenario to final scenario
        
        Args:
            - scenario: Scenario
            - scenario_name: Name for the scenario, used for plotting
        """
        if self.cga_population_manager.num_populations() > 0:
            return
        
        # Collect all scenarios in the tree, with the index being the tree level
        scenario_tree : List[Scenario] = [scenario]
        parent_id = scenario.parent_id
        while parent_id is not None:
            parent = self.get_generated_scenario_from_id(parent_id)
            scenario_tree.insert(0,parent)
            parent_id = parent.parent_id

        save_dir = os.path.join(self.dirs["figures"], f"{scenario_name}_tree")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        info_file_name = os.path.join(save_dir, "info.txt")

        self.log("info", f"Plotting scenario tree levels for '{scenario_name}'...")
        for idx in range(len(scenario_tree)):
            info = f"Level {idx} (Iter {scenario_tree[idx].id + 1}): Avg. Regret {scenario_tree[idx].avg_regret.value:.2f}"
            with open(info_file_name, "a") as f:
                f.write(info + "\n")
            self.log("inf", f"Plotting tree level {idx}...")
            self.plot_scene_from_scenario(scenario_tree[idx], os.path.join(save_dir, f"level_{idx}"))

    def analyze_scenario_snapshots(self, scenario: Scenario):
        """Analyze the snapshots from the given scenario
        
        Returns:
            - mean_astar_replan_angle: [deg]
            - max_astar_replan_angle: [deg]
            - mean_path_follower_angle: [deg]
            - max_path_follower_angle: [deg]
            - avg_loops_per_path: Average number of loops per path
            - weighted_obstacle_detection_distance: [m]
            - mean_image_proc_durations [sec]
            - max_image_proc_durations [sec]
            - mean_astar_replan_iterations 
            - max_astar_replan_iterations
            - mean_astar_replan_durations [sec]
            - max_astar_replan_durations [sec]
        """
        # Aggregate data across all outcmomes within a scenario
        astar_replan_angles = [] # [rad]
        path_follower_angles = [] # [rad]
        astar_obs_votes = []
        image_proc_durations = []
        astar_replan_iterations = []
        astar_replan_durations = []
        num_loops = 0
        for outcome in scenario.outcomes:
            # Get all A* angle from the A* snapshots
            for snapshot in outcome.astar_path_snapshots:
                if len(snapshot.opt_path) > 1:
                    euler = transforms3d.euler.quat2euler(snapshot.vehicle_orientation, 'rzyx')
                    vehicle_direction = np.array([np.cos(euler[0]), np.sin(euler[0])], np.float32)
                    initial_path_direction = snapshot.opt_path[1] - snapshot.opt_path[0]
                    initial_path_direction = initial_path_direction / fast_norm(initial_path_direction)

                    angle = np.arccos(np.clip(vehicle_direction.dot(initial_path_direction), -1., 1.))
                    if not np.isnan(angle):
                        astar_replan_angles.append(angle)

            # Get all angle diffs between vehicle position and tracking point from path follower snapshots
            for snapshot in outcome.path_follower_snapshots:
                euler = transforms3d.euler.quat2euler(snapshot.vehicle_orientation, 'rzyx')
                vehicle_direction = np.array([np.cos(euler[0]), np.sin(euler[0])], np.float32)
                tracking_direction = snapshot.tracking_waypoint - snapshot.vehicle_position # Here, both variables are 1-D numpy arrays of (x,y)
                tracking_direction = tracking_direction / fast_norm(tracking_direction)

                angle = np.arccos(np.clip(vehicle_direction.dot(tracking_direction), -1., 1.))
                if not np.isnan(angle): # We may get an occasional nan, not sure why, but let's just filter it out
                    path_follower_angles.append(angle)

            # Get critical info from A* snapshot summary
            image_proc_durations += outcome.astar_snapshots_summary.image_processing_duration
            astar_replan_iterations += outcome.astar_snapshots_summary.astar_iterations_per_replanning_step
            astar_replan_durations += outcome.astar_snapshots_summary.astar_replanning_step_duration

            # Check for loops in vehicle path
            # 1. Get cumulative arc length across vehicle path
            vehicle_path = outcome.vehicle_path[:,0:2] # Only care about xy
            vehicle_path_offset = vehicle_path.copy()
            vehicle_path_offset[1:] = vehicle_path[0:-1] # Offset array is [x1, x1, x2, x3, x4..., x(N-1)], where xi is a point
            arc_length_delta = fast_norm(vehicle_path - vehicle_path_offset, axis=1)
            arc_length = np.cumsum(arc_length_delta)
            
            # 2. Search
            ARC_SEPARATION = 25.
            PHYSICAL_SEPARATION = 5.
            ref_idx = 0
            while True:
                # Find closest point separated by certain distance along the trajectory arc
                search_idx = None
                for i in range(ref_idx, vehicle_path.shape[0]):
                    if arc_length[i] - arc_length[ref_idx] >= ARC_SEPARATION:
                        search_idx = i
                        break
                    
                if search_idx is None:
                    break
                
                search_vehicle_path = vehicle_path[search_idx:]
                search_distances = fast_norm(search_vehicle_path - vehicle_path[ref_idx], axis=1)
                argmin = np.argmin(search_distances)
                if search_distances[argmin] <= PHYSICAL_SEPARATION:
                    num_loops += 1
                    ref_idx = search_idx + argmin
                else:
                    # Try to advance ref_idx by 1 meter
                    found_new_ref_idx = False
                    for i in range(ref_idx, vehicle_path.shape[0]):
                        if arc_length[i] - arc_length[ref_idx] >= 1.:
                            ref_idx = i
                            found_new_ref_idx = True
                            break
                    if not found_new_ref_idx:
                        break

            astar_obs_votes.append(outcome.astar_path_snapshots[-1].obstacle_votes)

        # Store mean/max, and convert to degrees
        mean_astar_replan_angle = np.mean(astar_replan_angles) * 180./math.pi
        max_astar_replan_angle = np.max(astar_replan_angles) * 180./math.pi
        mean_path_follower_angle = np.mean(path_follower_angles) * 180./math.pi
        max_path_follower_angle = np.max(path_follower_angles) * 180./math.pi
        avg_loops_per_path = num_loops / len(scenario.outcomes)

        mean_image_proc_durations = np.mean(image_proc_durations)
        max_image_proc_durations = np.max(image_proc_durations)
        mean_astar_replan_iterations = np.mean(astar_replan_iterations)
        max_astar_replan_iterations = np.max(astar_replan_iterations)
        mean_astar_replan_durations = np.mean(astar_replan_durations)
        max_astar_replan_durations = np.max(astar_replan_durations)

        # Compute weighted distance of aggregate obstacle votes
        agg_obs_votes = np.sum(astar_obs_votes,axis=0)
        weighted_obstacle_detection_distance = 0.
        if np.sum(agg_obs_votes) > 0.:
            visible_obs_xy = self.scenario_builder.get_visible_ssa_locations(scenario.scene_description.ssa_array)

            # Get distance of each minor vertex to closest obstacle (vectorized method)
            minor_vertices = scenario.outcomes[0].astar_path_snapshots[0].minor_vertices
            vertex_distances = [fast_norm(minor_vertices - xy, axis=1).flatten() for xy in visible_obs_xy]
            vertex_distances_np = np.array(vertex_distances, dtype=np.float32)
            obstacle_distances = np.min(vertex_distances_np, axis=0)

            weights = agg_obs_votes / np.sum(agg_obs_votes) # Normalize
            weighted_obstacle_detection_distance = obstacle_distances.dot(weights)

        return mean_astar_replan_angle, max_astar_replan_angle, mean_path_follower_angle, max_path_follower_angle, avg_loops_per_path, weighted_obstacle_detection_distance,\
                mean_image_proc_durations, max_image_proc_durations, mean_astar_replan_iterations, max_astar_replan_iterations, mean_astar_replan_durations, max_astar_replan_durations

    def analyze_generated_scenarios_snapshots(self):
        """Analyze the generated scenarios snapshots and make relevant plots"""
        self.log("info", "Analyzing scenario snapshots...")

        self.log("info", F"Verifying all scenarios have required data fields...")
        for id in range(self.scenario_regret_manager.num_scenarios()):
            scenario = self.get_generated_scenario_from_id(id)
            for j,outcome in enumerate(scenario.outcomes):
                if outcome.astar_path_snapshots is None:
                    self.log("warn", f"Scenario {id} outcome {j} has no astar path snapshots.")
                if outcome.astar_snapshots_summary is None:
                    self.log("warn", f"Scenario {id} outcome {j} has no astar snapshots summary.")
                if outcome.path_follower_snapshots is None:
                    self.log("warn", f"Scenario {id} outcome {j} has no path follower snapshots.")
                if outcome.path_follower_snapshots_summary is None:
                    self.log("warn", f"Scenario {id} outcome {j} has no path follower snapshots summary.")

        self.log("info", F"Gathering necesary data from scenarios...")
        
        counter = 0
        self.scenario_analysis = ScenarioAnalysis()
        for id in range(self.scenario_regret_manager.num_scenarios()):
            scenario = self.get_generated_scenario_from_id(id)
            
            (mean_astar_replan_angle, 
             max_astar_replan_angle, 
             mean_path_follower_angle, 
             max_path_follower_angle, 
             avg_loops_per_path,
             weighted_obs_detection_distance,
             mean_image_proc_durations, 
             max_image_proc_durations, 
             mean_astar_replan_iterations, 
             max_astar_replan_iterations, 
             mean_astar_replan_durations, 
             max_astar_replan_durations) = self.analyze_scenario_snapshots(scenario)
            
            # Store values
            self.scenario_analysis.mean_astar_replan_angle.append(mean_astar_replan_angle)
            self.scenario_analysis.max_astar_replan_angle.append(max_astar_replan_angle)
            self.scenario_analysis.mean_path_follower_angle.append(mean_path_follower_angle)
            self.scenario_analysis.max_path_follower_angle.append(max_path_follower_angle)
            self.scenario_analysis.avg_loops_per_path.append(avg_loops_per_path)
            self.scenario_analysis.weighted_obstacle_detection_distance.append(weighted_obs_detection_distance)

            self.scenario_analysis.mean_image_processing_durations.append(mean_image_proc_durations)
            self.scenario_analysis.max_image_processing_durations.append(max_image_proc_durations)
            self.scenario_analysis.mean_astar_replan_iterations.append(mean_astar_replan_iterations)
            self.scenario_analysis.max_astar_replan_iterations.append(max_astar_replan_iterations)
            self.scenario_analysis.mean_astar_replan_durations.append(mean_astar_replan_durations)
            self.scenario_analysis.max_astar_replan_durations.append(max_astar_replan_durations)

            counter += 1
            if (counter % 200 == 0):
                self.log("info", f"Progress: {counter} / {self.scenario_regret_manager.num_scenarios()}")

        with open(self.scenario_analysis_filepath, 'wb') as f:
            pickle.dump(self.scenario_analysis, f, pickle.HIGHEST_PROTOCOL)
        self.log("info", f"Saved scenario analysis to: {self.scenario_analysis_filepath}")

        # self.plot_scenario_analysis() # This takes too long, commenting out

    def plot_scenario_analysis(self):
        self.log("info", f"Plotting scenario analysis...")
        sorted_regrets = []
        sorted_ids = []
        for id,regret in self.scenario_regret_manager.sorted_regrets:
            sorted_regrets.append(regret)
            sorted_ids.append(id)

        save_dir = os.path.join(self.dirs["figures"], "analysis")
        os.makedirs(save_dir, exist_ok = True)
        
        # Helper lambda functions
        get_sorted_array = lambda arr, idxs: [arr[i] for i in idxs]
        stats_to_str = lambda array: f"Min = {np.min(array):.2f} / Mean = {np.mean(array):.2f} / Max = {np.max(array):.2f}"

        sorted_mean_astar_replan_angle = get_sorted_array(self.scenario_analysis.mean_astar_replan_angle, sorted_ids)
        sorted_max_astar_replan_angle = get_sorted_array(self.scenario_analysis.max_astar_replan_angle, sorted_ids)
        sorted_mean_path_follower_angle = get_sorted_array(self.scenario_analysis.mean_path_follower_angle, sorted_ids)
        sorted_max_path_follower_angle = get_sorted_array(self.scenario_analysis.max_path_follower_angle, sorted_ids)
        sorted_avg_loops_per_path = get_sorted_array(self.scenario_analysis.avg_loops_per_path, sorted_ids)
        sorted_weighted_obstacle_detection_distance = get_sorted_array(self.scenario_analysis.weighted_obstacle_detection_distance, sorted_ids)

        sorted_mean_image_processing_durations = get_sorted_array(self.scenario_analysis.mean_image_processing_durations, sorted_ids)
        sorted_max_image_processing_durations = get_sorted_array(self.scenario_analysis.max_image_processing_durations, sorted_ids)
        sorted_mean_astar_replan_iterations = get_sorted_array(self.scenario_analysis.mean_astar_replan_iterations, sorted_ids)
        sorted_max_astar_replan_iterations = get_sorted_array(self.scenario_analysis.max_astar_replan_iterations, sorted_ids)
        sorted_mean_astar_replan_durations = get_sorted_array(self.scenario_analysis.mean_astar_replan_durations, sorted_ids)
        sorted_max_astar_replan_durations = get_sorted_array(self.scenario_analysis.max_astar_replan_durations, sorted_ids)

        analysis_summary_filepath = os.path.join(save_dir, "scenario_analysis_summary.txt")
        with open(analysis_summary_filepath, "w") as f:
            f.write(f"Scenario Analysis Summary:\n")

            f.write(f"Mean A* Replan Angle [deg]: {stats_to_str(sorted_mean_astar_replan_angle)} \n")
            f.write(f"Max A* Replan Angle [deg]: {stats_to_str(sorted_max_astar_replan_angle)} \n")
            f.write(f"Mean Path Follower Angle [deg]: {stats_to_str(sorted_mean_path_follower_angle)} \n")
            f.write(f"Max Path Follower Angle [deg]: {stats_to_str(sorted_max_path_follower_angle)} \n")
            f.write(f"Avg. Loops Per Path: {stats_to_str(sorted_avg_loops_per_path)} \n")
            f.write(f"Weighted Obstacle Detection Distance [m]: {stats_to_str(sorted_weighted_obstacle_detection_distance)} \n")

            f.write(f"Mean Image Processing Durations [s]: {stats_to_str(sorted_mean_image_processing_durations)} \n")
            f.write(f"Max Image Processing Durations [s]: {stats_to_str(sorted_max_image_processing_durations)} \n")
            f.write(f"Mean A* Replan Iterations: {stats_to_str(sorted_mean_astar_replan_iterations)} \n")
            f.write(f"Max A* Replan Iterations: {stats_to_str(sorted_max_astar_replan_iterations)} \n")
            f.write(f"Mean A* Replan Durations [s]: {stats_to_str(sorted_mean_astar_replan_durations)} \n")
            f.write(f"Max A* Replan Durations [s]: {stats_to_str(sorted_max_astar_replan_durations)} \n")

        rankings = np.arange(self.scenario_regret_manager.num_scenarios()) + 1
        fontsize=16
        regret_bar_width = 0.001
        std_bar_width = 0.65

        def plot_mean_max(xaxis, mean_data, mean_label, max_data, max_label, ymajor_locator, file_path):
            assert xaxis == "ranking" or xaxis == "regret"
            fig, ax = plt.subplots(2,1,num=1)
            
            if xaxis == "ranking":
                # Avg
                ax[0].bar(rankings, mean_data, width=std_bar_width, color="black", edgecolor="black", linewidth=0.)
                ax[0].set_xlabel(f"Scenario Ranking Index", fontsize=fontsize)
                # Max
                ax[1].bar(rankings, max_data, width=std_bar_width, color="black", edgecolor="black", linewidth=0.)
                ax[1].set_xlabel(f"Scenario Ranking Index", fontsize=fontsize)
            elif xaxis == "regret":
                # Avg
                ax[0].bar(sorted_regrets, mean_data, width=regret_bar_width, color="black", edgecolor="black", linewidth=0.)
                ax[0].set_xlabel(f"Regret", fontsize=fontsize)
                # Max
                ax[0].bar(sorted_regrets, max_data, width=regret_bar_width, color="black", edgecolor="black", linewidth=0.)
                ax[0].set_xlabel(f"Regret", fontsize=fontsize)

            ax[0].set_ylabel(mean_label, fontsize=fontsize)
            ax[1].set_ylabel(max_label, fontsize=fontsize)
            if ymajor_locator > 0: # If negative, let matplotlib figure it out
                ax[0].yaxis.set_major_locator(ticker.MultipleLocator(ymajor_locator))
                ax[1].yaxis.set_major_locator(ticker.MultipleLocator(ymajor_locator))
            ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
            ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
            
            fig.tight_layout()
            fig.savefig(file_path)
            fig.clf()
            
        # A* Replan Angle
        file_path = os.path.join(save_dir, f"astar_replanning_angle_vs_ranking.pdf")
        plot_mean_max("ranking", sorted_mean_astar_replan_angle, f"Avg. Replan\nAngle [deg.]", sorted_max_astar_replan_angle, f"Max Replan\nAngle [deg.]", 50, file_path)
        file_path = os.path.join(save_dir, f"astar_replanning_angle_vs_regret.pdf")
        plot_mean_max("regret", sorted_mean_astar_replan_angle, f"Avg. Replan\nAngle [deg.]", sorted_max_astar_replan_angle, f"Max Replan\nAngle [deg.]", 50, file_path)
        
        # Path Planner Angle
        file_path = os.path.join(save_dir, f"tracking_point_angle_vs_ranking.pdf")
        plot_mean_max("ranking", sorted_mean_path_follower_angle, f"Avg. Tracking\nAngle [deg.]", sorted_max_path_follower_angle, f"Max Tracking\nAngle [deg.]", 50, file_path)
        file_path = os.path.join(save_dir, f"tracking_point_angle_vs_regret.pdf")
        plot_mean_max("regret", sorted_mean_path_follower_angle, f"Avg. Tracking\nAngle [deg.]", sorted_max_path_follower_angle, f"Max Tracking\nAngle [deg.]", 50, file_path)
        
        # Avg number of loops per path
        fig, ax = plt.subplots(2,1,num=1)
        # vs. ranking
        ax[0].bar(rankings, sorted_avg_loops_per_path, width=std_bar_width, color="black", edgecolor="black", linewidth=0.)
        ax[0].set_xlabel(f"Scenario Ranking Index", fontsize=fontsize)
        ax[0].set_ylabel(f"Avg. Path Loops", fontsize=fontsize)
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
        # vs. regret
        ax[1].bar(sorted_regrets, sorted_avg_loops_per_path, width=regret_bar_width, color="black", edgecolor="black", linewidth=0.)
        ax[1].set_xlabel(f"Regret", fontsize=fontsize)
        ax[1].set_ylabel(f"Avg. Path Loops", fontsize=fontsize) # Avg. Number of Loops per Path
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
        fig.tight_layout()
        file_path = os.path.join(save_dir, f"avg_loops_per_path.pdf")
        fig.savefig(file_path)
        fig.clf()
        
        # Weighted Obstacle Detection Distance vs ranking
        if np.max(sorted_weighted_obstacle_detection_distance) <= 20.:
            ytick_spacing = 5
        elif np.max(sorted_weighted_obstacle_detection_distance) <= 50.:
            ytick_spacing = 10
        else:
            ytick_spacing = 20
        fig, ax = plt.subplots(2,1,num=1)
        # vs. ranking
        ax[0].bar(rankings, sorted_weighted_obstacle_detection_distance, width=std_bar_width, color="black", edgecolor="black", linewidth=0.)
        ax[0].set_xlabel(f"Scenario Ranking Index", fontsize=fontsize)
        ax[0].set_ylabel(f"Detection\nDistance [m]", fontsize=fontsize)
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
        ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
        # vs. regret
        ax[1].bar(sorted_regrets, sorted_weighted_obstacle_detection_distance, width=regret_bar_width, color="black", edgecolor="black", linewidth=0.)
        ax[1].set_xlabel(f"Regret", fontsize=fontsize)
        ax[1].set_ylabel(f"Detection\nDistance [m]", fontsize=fontsize) # Weighted Obstacle\nDetection Distance [m]
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
        ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
        fig.tight_layout()
        file_path = os.path.join(save_dir, f"weighted_obstacle_detection_distance.pdf")
        fig.savefig(file_path)
        fig.clf()

        # Image Proc Time
        file_path = os.path.join(save_dir, f"image_processing_time_vs_ranking.pdf")
        plot_mean_max("ranking", sorted_mean_image_processing_durations, f"Avg. Image\nProc. Time [s]", sorted_max_image_processing_durations, f"Max Image\nProc. Time [s]", -1, file_path)
        file_path = os.path.join(save_dir, f"image_processing_time_vs_regret.pdf")
        plot_mean_max("regret", sorted_mean_image_processing_durations, f"Avg. Image\nProc. Time [s]", sorted_max_image_processing_durations, f"Max Image\nProc. Time [s]", -1, file_path)
        
        # A* Replan Time
        file_path = os.path.join(save_dir, f"astar_replan_time_vs_ranking.pdf")
        plot_mean_max("ranking", sorted_mean_astar_replan_durations, f"Avg. Replan\nTime [s]", sorted_max_astar_replan_durations, f"Max Replan\nTime [s]", -1, file_path)
        file_path = os.path.join(save_dir, f"astar_replan_time_vs_regret.pdf")
        plot_mean_max("regret", sorted_mean_astar_replan_durations, f"Avg. Replan\nTime [s]", sorted_max_astar_replan_durations, f"Max Rreplan\nTime [s]", -1, file_path)
        
        # A* Replan Iterations
        file_path = os.path.join(save_dir, f"astar_replan_iterations_vs_ranking.pdf")
        plot_mean_max("ranking", sorted_mean_astar_replan_iterations, f"Avg. Replan\nIterations", sorted_max_astar_replan_iterations, f"Max Replan\nIterations", -1, file_path)
        file_path = os.path.join(save_dir, f"astar_replan_iterations_vs_regret.pdf")
        plot_mean_max("regret", sorted_mean_astar_replan_iterations, f"Avg. Replan\nIterations", sorted_max_astar_replan_iterations, f"Max Rreplan\nIterations", -1, file_path)
        
    def reanalyze_replay_scenarios(self):
        """Reanalyze all replay scenarios"""
        self.log("info", f"Reanalyzing replay scenarios...")
        replay_dirs = [f.path for f in os.scandir(self.dirs["replay"]) if f.is_dir()]
        for rd in replay_dirs:
            # self.log("info", f"Reanalyzing replays in: replay/{rd.split('/')[-1]}")
            sub_dirs = [f.path for f in os.scandir(rd) if f.is_dir()]
            for sd in sub_dirs:
                self.log("info", f"Reanalyzing replays in: {sd}")

                # Remove old files in scenario directory?
                files = glob.glob(os.path.join(sd, "*.pdf")) + glob.glob(os.path.join(sd, "*.png"))
                for file in files:
                    os.remove(file)

                self.analyze_replay_scenario(sd, scenario=None)

    def plot_group_summary(self, group_summary_dict: Dict, figure_prefix: str, num_legend_cols: int = 3, legend_fontsize: float = 12):
        self.log("info", "Making group summary plots...")

        # Load all scenario regret manager data in same structure as group names
        group_regrets = {}
        group_regrets_smoothed = {}
        group_max_regrets = {}
        for group_name, (folders,color) in group_summary_dict.items():
            regrets = []
            regrets_smoothed = []
            max_regrets = []
            for name in folders:
                self.log("info", f"Loading scenario regret manager from folder {name}")
                srm_path = os.path.join(self.dirs["outer"], name, "data", "scenario_regret_manager.pkl")
                srm, loaded = self.load_scenario_regret_manager(srm_path)
                if loaded:
                    regrets.append(list(srm.scenario_regrets.values()))
                    regrets_smoothed.append(get_smoothing_avg(list(srm.scenario_regrets.values()), 100))
                    max_regrets.append(np.array(srm.max_regret_over_time)[:,1])
                else:
                    raise ValueError(f"Could not load scenario regret manager from path: {srm_path}")
            group_regrets[group_name] = np.array(regrets)
            group_regrets_smoothed[group_name] = np.array(regrets_smoothed)
            group_max_regrets[group_name] = np.array(max_regrets)

        fontsize = 12
        # legend_fontsize = 12

        # Regret vs. Iterations
        fig, ax = plt.subplots(num=1)
        ax.set_xlabel(f"Iterations", fontsize=fontsize)
        ax.set_ylabel(f"Regret", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.axvline(self.num_base_env, linestyle='--', linewidth=1, color="black", label="Num. Base\nScenarios")
        if self.anneal_duration > 0:
            ax.axvline(self.anneal_duration + self.num_base_env, linestyle='--', linewidth=1, color="magenta", label="Annealing Ends")

        # Max Regret vs. Iterations
        fig2, ax2 = plt.subplots(num=2)
        ax2.set_xlabel(f"Iterations", fontsize=fontsize)
        ax2.set_ylabel(f"Max Regret", fontsize=fontsize)
        ax2.tick_params(axis="both", which="major", labelsize=fontsize)
        ax2.axvline(self.num_base_env, linestyle='--', linewidth=1, color="black", label="Num. Base\nScenarios")
        if self.anneal_duration > 0:
            ax2.axvline(self.anneal_duration + self.num_base_env, linestyle='--', linewidth=1, color="magenta", label="Annealing Ends")

        # Side-by-Side (w/ standard error)
        ar3 = 16/9
        width3 = 6.4
        fig3, ax3 = plt.subplots(ncols=2,num=3, figsize=(width3, width3/ar3))
        ax3[0].set_xlabel(f"Iterations", fontsize=fontsize)
        ax3[0].set_ylabel(f"Regret", fontsize=fontsize)
        ax3[0].tick_params(axis="both", which="major", labelsize=fontsize)
        ax3[0].axvline(self.num_base_env, linestyle='--', linewidth=1, color="black", label="Num. Base\nScenarios")
        
        ax3[1].set_xlabel(f"Iterations", fontsize=fontsize)
        ax3[1].set_ylabel(f"Max Regret", fontsize=fontsize)
        ax3[1].tick_params(axis="both", which="major", labelsize=fontsize)
        ax3[1].axvline(self.num_base_env, linestyle='--', linewidth=1, color="black", label="Num. Base\nScenarios")
        if self.anneal_duration > 0:
            ax3[0].axvline(self.anneal_duration + self.num_base_env, linestyle='--', linewidth=1, color="magenta", label="Annealing Ends")
            ax3[1].axvline(self.anneal_duration + self.num_base_env, linestyle='--', linewidth=1, color="magenta", label="Annealing Ends")
        
        # Side-by-Side (avg. only, no std. error)
        fig4, ax4 = plt.subplots(ncols=2,num=4, figsize=(width3, width3/ar3))
        ax4[0].set_xlabel(f"Iterations", fontsize=fontsize)
        ax4[0].set_ylabel(f"Regret", fontsize=fontsize)
        ax4[0].tick_params(axis="both", which="major", labelsize=fontsize)
        ax4[0].axvline(self.num_base_env, linestyle='--', linewidth=1, color="black", label="Num. Base\nScenarios")
        
        ax4[1].set_xlabel(f"Iterations", fontsize=fontsize)
        ax4[1].set_ylabel(f"Max Regret", fontsize=fontsize)
        ax4[1].tick_params(axis="both", which="major", labelsize=fontsize)
        ax4[1].axvline(self.num_base_env, linestyle='--', linewidth=1, color="black", label="Num. Base\nScenarios")
        if self.anneal_duration > 0:
            ax4[0].axvline(self.anneal_duration + self.num_base_env, linestyle='--', linewidth=1, color="magenta", label="Annealing Ends")
            ax4[1].axvline(self.anneal_duration + self.num_base_env, linestyle='--', linewidth=1, color="magenta", label="Annealing Ends")

        xrange = 1+np.arange(self.num_env_to_generate)
        for group_name in group_summary_dict.keys():
            group_color = group_summary_dict[group_name][-1]

            iter_mean = np.mean(group_regrets_smoothed[group_name], axis=0)
            iter_std = np.std(group_regrets_smoothed[group_name], axis=0)
            max_mean = np.mean(group_max_regrets[group_name], axis=0)
            max_std = np.std(group_max_regrets[group_name], axis=0)
            z = 1. # 1.96 for 95% CI, 1 for std error

            ax.fill_between(xrange, iter_mean - z*iter_std, iter_mean + z*iter_std, edgecolor=group_color, facecolor=group_color, alpha=0.2)
            ax.plot(xrange, iter_mean, linewidth=1, color=group_color, label=group_name)
            ax3[0].fill_between(xrange, iter_mean - z*iter_std, iter_mean + z*iter_std, edgecolor=group_color, facecolor=group_color, alpha=0.2)
            ax3[0].plot(xrange, iter_mean, linewidth=1, color=group_color, label=group_name)
            ax4[0].plot(xrange, iter_mean, linewidth=1, color=group_color, label=group_name)
            
            ax2.fill_between(xrange, max_mean - z*max_std, max_mean + z*max_std, edgecolor=group_color, facecolor=group_color, alpha=0.2)
            ax2.plot(xrange, max_mean, linewidth=1, color=group_color, label=group_name)
            ax3[1].fill_between(xrange, max_mean - z*max_std, max_mean + z*max_std, edgecolor=group_color, facecolor=group_color, alpha=0.2)
            ax3[1].plot(xrange, max_mean, linewidth=1, color=group_color, label=group_name)
            ax4[1].plot(xrange, max_mean, linewidth=1, color=group_color, label=group_name)

        handles, labels = ax.get_legend_handles_labels()

        fig.tight_layout(pad=0.5)
        fig.legend(handles, labels, loc='lower center', ncol=num_legend_cols, bbox_to_anchor=(.5, 1), fontsize=legend_fontsize)
        fig.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_regret_vs_iterations.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_regret_vs_iterations.png"), bbox_inches="tight")
        
        fig2.tight_layout(pad=0.5)
        fig2.legend(handles, labels, loc='lower center', ncol=num_legend_cols, bbox_to_anchor=(.5, 1), fontsize=legend_fontsize)
        fig2.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_max_regret_vs_iterations.pdf"), bbox_inches="tight")
        fig2.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_max_regret_vs_iterations.png"), bbox_inches="tight")
        
        fig3.tight_layout()
        fig3.legend(handles, labels, loc='lower center', ncol=num_legend_cols, bbox_to_anchor=(.5, 1.), fontsize=legend_fontsize)
        fig3.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_side_by_side_regret_vs_iterations.pdf"), bbox_inches="tight")
        fig3.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_side_by_side_regret_vs_iterations.png"), bbox_inches="tight")
        # fig3.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_side_by_side_regret_vs_iterations.pdf"), bbox_extra_artists=(legend3,), bbox_inches="tight")
        
        fig4.tight_layout()
        fig4.legend(handles, labels, loc='lower center', ncol=num_legend_cols, bbox_to_anchor=(.5, 1.), fontsize=legend_fontsize)
        fig4.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_side_by_side_regret_vs_iterations_avg_only.pdf"), bbox_inches="tight")
        fig4.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_side_by_side_regret_vs_iterations_avg_only.png"), bbox_inches="tight")
        # fig4.savefig(os.path.join(self.dirs["group_summary"], f"{figure_prefix}_side_by_side_regret_vs_iterations.pdf"), bbox_extra_artists=(legend3,), bbox_inches="tight")
        
        fig.clf()
        fig2.clf()
        fig3.clf()
        fig4.clf()
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

    def plot_cga_group_summary(self, group_summary_dict: Dict):
        self.log("info", "Making CGA group summary plots...")

        # Load all CGA population manager data in same structure as group names
        group_cga_regrets = {}
        group_cga_max_regrets = {}
        for group_name, (folders,color) in group_summary_dict.items():
            regrets = []
            max_regrets = []
            for name in folders:
                self.log("info", f"Loading CGA population manager from folder {name}")
                cga_path = os.path.join(self.dirs["outer"], name, "data", "cga_population_manager.pkl")
                cpm, loaded = self.load_cga_population_manager(cga_path)
                if loaded:
                    regrets.append(cpm.avg_population_regret)
                    max_regrets.append(cpm.max_population_regret)
                else:
                    raise ValueError(f"Could not load CGA population manager from path: {cga_path}")
            group_cga_regrets[group_name] = np.array(regrets)
            group_cga_max_regrets[group_name] = np.array(max_regrets)

        # Load all scenario regret manager data in same structure as group names
        group_regrets = {}
        for group_name, (folders,color) in group_summary_dict.items():
            regrets = []
            for name in folders:
                self.log("info", f"Loading scenario regret manager from folder {name}")
                srm_path = os.path.join(self.dirs["outer"], name, "data", "scenario_regret_manager.pkl")
                srm, loaded = self.load_scenario_regret_manager(srm_path)
                if loaded:
                    regrets.append(list(srm.scenario_regrets.values()))
                else:
                    raise ValueError(f"Could not load scenario regret manager from path: {srm_path}")
            group_regrets[group_name] = np.array(regrets)

        fontsize = 12
        legend_fontsize = 12

        ar3 = 16/9
        width3 = 6.4
        fig3, ax3 = plt.subplots(ncols=2,num=3, figsize=(width3, width3/ar3))
        ax3[0].set_xlabel(f"Population", fontsize=fontsize)
        ax3[0].set_ylabel(f"Average Regret", fontsize=fontsize)
        ax3[0].tick_params(axis="both", which="major", labelsize=fontsize)
        
        ax3[1].set_xlabel(f"Population", fontsize=fontsize)
        ax3[1].set_ylabel(f"Max Regret", fontsize=fontsize)
        ax3[1].tick_params(axis="both", which="major", labelsize=fontsize)

        xrange = 1+np.arange(self.cga_num_populations)
        for group_name in group_summary_dict.keys():
            group_color = group_summary_dict[group_name][-1]

            pop_mean = np.mean(group_cga_regrets[group_name], axis=0)
            pop_std = np.std(group_cga_regrets[group_name], axis=0)
            max_mean = np.mean(group_cga_max_regrets[group_name], axis=0)
            max_std = np.std(group_cga_max_regrets[group_name], axis=0)
            z = 1. # 1.96 for 95% CI, 1 for std error

            ax3[0].fill_between(xrange, pop_mean - z*pop_std, pop_mean + z*pop_std, edgecolor=group_color, facecolor=group_color, alpha=0.2)
            ax3[0].plot(xrange, pop_mean, linewidth=1, color=group_color, label=group_name)
            
            ax3[1].fill_between(xrange, max_mean - z*max_std, max_mean + z*max_std, edgecolor=group_color, facecolor=group_color, alpha=0.2)
            ax3[1].plot(xrange, max_mean, linewidth=1, color=group_color, label=group_name)

        handles, labels = ax3[0].get_legend_handles_labels()

        fig3.tight_layout()
        fig3.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, 1.), fontsize=legend_fontsize)
        fig3.savefig(os.path.join(self.dirs["group_summary"], f"cga_side_by_side_regret_vs_populations.pdf"), bbox_inches="tight")
        fig3.savefig(os.path.join(self.dirs["group_summary"], f"cga_side_by_side_regret_vs_populations.png"), bbox_inches="tight")

        fig3.clf()
        plt.close(fig3)

    def group_summary_logging(self, group_summary_dict: Dict, regret_thresholds: Tuple[float], filename_prefix: str):
        self.log("info", "Making group summary logs...")

        # Load all scenario regret manager data in same structure as group names
        group_regrets = {}
        group_failure_rates = {}
        for group_name, folders in group_summary_dict.items():
            regrets = []
            failure_rates = []
            for name in folders:
                self.log("info", f"Loading scenario regret manager from folder {name}")
                srm_path = os.path.join(self.dirs["outer"], name, "data", "scenario_regret_manager.pkl")
                srm, loaded = self.load_scenario_regret_manager(srm_path)
                if loaded:
                    regrets.append(list(srm.scenario_regrets.values()))
                    failure_rates.append(list(srm.failure_rate.values()))
                else:
                    raise ValueError(f"Could not load scenario regret manager from path: {srm_path}")
            group_regrets[group_name] = np.array(regrets)
            group_failure_rates[group_name] = np.array(failure_rates)

        regret_thresholds_file = os.path.join(self.dirs["group_summary"], f"{filename_prefix}_regret_thresholds_group_summary.txt")
        with open(regret_thresholds_file, "w") as f:
            f.write(f"Scenarios per group whose regret is >= to the following thresholds: {regret_thresholds}\n")

        failure_rate_file = os.path.join(self.dirs["group_summary"], f"{filename_prefix}_failure_rate_group_summary.txt")
        with open(failure_rate_file, "w") as f:
            f.write(f"Fraction of scenario outcomes resulting in vehicle failure\n")

        for group_name in group_summary_dict.keys():
            with open(regret_thresholds_file, "a") as f:
                f.write(f"{group_name}:\n")
                n1,n2 = group_regrets[group_name].shape
                f.write(f"\t- Scenario Count: {group_regrets[group_name].size} ({n1} Experiments with {n2} Scenarios Each)\n")

            # Number of scenarios with regret above certain thresholds
            for t in regret_thresholds:
                exceed_threshold = group_regrets[group_name] >= t
                count = np.count_nonzero(exceed_threshold)
                fraction = count/group_regrets[group_name].size
                
                # Find how long it took to exceed each threshold
                iters = []
                for i in range(exceed_threshold.shape[0]):
                    for j in range(exceed_threshold.shape[1]):
                        if exceed_threshold[i,j]:
                            iters.append(j+1)
                            break
                num_iters_to_exceed = np.mean(iters)
                exp_frac = len(iters) / n1

                with open(regret_thresholds_file, "a") as f:
                    f.write(f"\t- Threshold {t}: Scenario Count {count}, Scenario Fraction {fraction:.6f}, Experiment Count {len(iters)} / {n1}, Experiment Fraction {exp_frac:.6f}, Num. Iters to Reach (for the corresponding experiments) {num_iters_to_exceed:.1f} \n")

            # Find maximum regret achieved
            max_regrets = np.max(group_regrets[group_name],axis=-1)
            arg_max = np.argmax(max_regrets)
            max_folder = group_summary_dict[group_name][arg_max]
            with open(regret_thresholds_file, "a") as f:
                f.write(f"\t- Maximum Regret Achieved: {max_regrets[arg_max]:.6f} (Experiment Index {arg_max}, Folder \"{max_folder}\") \n")

            # Vehicle failure rate
            with open(failure_rate_file, "a") as f:
                f.write(f"{group_name}:\n")
                f.write(f"\t- Total Scenarios: {group_regrets[group_name].size}\n")
                f.write(f"\t- Vehicle Failure Rate: {np.mean(group_failure_rates[group_name]):.6f}\n")

    def get_extreme_path_values(self, vehicle_paths: List[np.float32]):
        """Get the min/max x,y values from the list of vehicle paths

        Args:
            - vehicle_paths: The list of vehicle paths
        
        Returns:
            - min_path_x, max_path_x, min_path_y, max_path_y 
        """
        min_path_x = math.inf
        min_path_y = math.inf
        max_path_x = -math.inf
        max_path_y = -math.inf
        for vehicle_path in vehicle_paths:
            path_x = vehicle_path[:,0].tolist()
            path_y = vehicle_path[:,1].tolist()

            min_path_x = min(min_path_x, min(path_x))
            min_path_y = min(min_path_y, min(path_y))
            max_path_x = max(max_path_x, max(path_x))
            max_path_y = max(max_path_y, max(path_y))
        
        return min_path_x, max_path_x, min_path_y, max_path_y

    def create_scene_capture_mosaic(self, 
                                    scenario: Scenario, 
                                    format: str = "landscape", 
                                    fig_width: float = None, 
                                    base_resolution: int = None,
                                    b_use_annotated_side_images: bool = False, 
                                    other_vehicle_paths: List[np.float32] = None):
        scene_capture_dir = os.path.join(self.dirs["scene_captures"], f"scenario_{scenario.id}")
        ortho_aerial_files = glob.glob(os.path.join(scene_capture_dir, "ortho_aerial_pad*"))
        if len(ortho_aerial_files) == 0:
            self.log("warn", f"Scenario ID {scenario.id} does not have any orthographic aerial scene captures")
            return None, None
        
        # Extract padding values
        padding = []
        for f in ortho_aerial_files:
            split1 = f.split("ortho_aerial_pad") # Returns something like ["file_prefix/scenario_XX", "Y.png"]
            if "_annotated.png" in f:
                split2 = split1[-1].split("_annotated.png")
            else:
                split2 = split1[-1].split(".png")

            val = int(split2[0])
            if val not in padding:
                padding.append(val)

        if other_vehicle_paths is None:
            min_path_x, max_path_x, min_path_y, max_path_y = self.get_extreme_path_values(scenario.get_vehicle_paths())
        else:
            min_path_x, max_path_x, min_path_y, max_path_y = self.get_extreme_path_values(scenario.get_vehicle_paths() + other_vehicle_paths)
        min_val = min(min_path_x, min_path_y)
        max_val = max(max_path_x, max_path_y)

        # Get maximum padding needed to show all vehicle paths
        sorted_padding = np.sort(padding).tolist() # Increasing order
        padding_requirements = []
        for p in sorted_padding:
            if min_val >= 0. - p:
                padding_requirements.append(p)
                break
        for p in sorted_padding:
            if max_val <= self.scenario_builder.landscape_nominal_size + p:
                padding_requirements.append(p)
                break
        
        # If one of the paths goes outside the image with largest padding available, then we will use that image
        if min_val < 0. - padding[-1] or max_val > self.scenario_builder.landscape_nominal_size + p:
            padding_requirements.append(padding[-1])
        
        ortho_padding = max(padding_requirements)

        ortho_aerial_image = cv2.cvtColor(cv2.imread(os.path.join(scene_capture_dir, f"ortho_aerial_pad{ortho_padding}.png")), cv2.COLOR_BGR2RGB)
        if b_use_annotated_side_images:
            front_aerial_image = cv2.cvtColor(cv2.imread(os.path.join(scene_capture_dir, f"perspective_front_aerial_45_annotated.png")), cv2.COLOR_BGR2RGB)
            vehicle_start_image = cv2.cvtColor(cv2.imread(os.path.join(scene_capture_dir, f"perspective_vehicle_start_rear_aerial_annotated.png")), cv2.COLOR_BGR2RGB)
        else:
            front_aerial_image = cv2.cvtColor(cv2.imread(os.path.join(scene_capture_dir, f"perspective_front_aerial_45.png")), cv2.COLOR_BGR2RGB)
            vehicle_start_image = cv2.cvtColor(cv2.imread(os.path.join(scene_capture_dir, f"perspective_vehicle_start_rear_aerial.png")), cv2.COLOR_BGR2RGB)

        if base_resolution is not None:
            s1 = base_resolution
            # s2 = int(base_resolution/2)
            ortho_aerial_image = cv2.resize(ortho_aerial_image, (s1,s1))
            front_aerial_image = cv2.resize(front_aerial_image, (s1,s1))
            vehicle_start_image = cv2.resize(vehicle_start_image, (s1,s1))

        size = self.scenario_builder.landscape_nominal_size
        extent = (0.-ortho_padding, size+ortho_padding, 0.-ortho_padding, size+ortho_padding)

        correction = 0.9935 # Strangely, without num=1, using 0.9935*desired_aspect_ratio reduces any slight white spacing between the image tiles
        # Use large figure ID number to prevent others from using this figure size
        if format == "landscape":
            fig, ax = plt.subplot_mosaic([["ortho_aerial", "front_aerial"], ["ortho_aerial", "vehicle_start"]], #num=100,
                                        gridspec_kw={"width_ratios": [2.,1.], "height_ratios": [1.,1.]})
            if fig_width is not None:
                fig.set_figwidth(fig_width)
            fig.set_figheight(fig.get_figwidth()/(1.5)*correction) 

        elif format == "portrait":
            fig, ax = plt.subplot_mosaic([["front_aerial", "vehicle_start"], ["ortho_aerial", "ortho_aerial"]], #num=100,
                                        gridspec_kw={"width_ratios": [1.,1.], "height_ratios": [1.,2.]})
            if fig_width is not None:
                fig.set_figwidth(fig_width)
            fig.set_figheight(fig.get_figwidth()/((2./3.)*correction))

        ax["ortho_aerial"].imshow(ortho_aerial_image, extent=extent, aspect="equal", interpolation="none")
        # ax["ortho_aerial"].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax["ortho_aerial"].yaxis.set_major_locator(ticker.MultipleLocator(size))
        ax["ortho_aerial"].xaxis.set_major_locator(ticker.MultipleLocator(size))
        ax["ortho_aerial"].set_xlim([extent[0], extent[1]])
        ax["ortho_aerial"].set_ylim([extent[0], extent[1]])

        ax["front_aerial"].imshow(front_aerial_image, extent=extent, aspect="equal", interpolation="none")
        ax["front_aerial"].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax["vehicle_start"].imshow(vehicle_start_image, extent=extent, aspect="equal", interpolation="none")
        ax["vehicle_start"].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        return fig, ax

    def make_base_scene_plot(self, 
                            scenario: Scenario, 
                            format: str = "landscape", 
                            fig_width: float = None, 
                            fontsize: float = 12.,
                            marker_scale: float = 1.,
                            base_resolution: int = None,
                            b_use_annotated_side_images: bool = False, 
                            other_vehicle_paths: List[np.float32] = None):
        # Always try to make a scene capture mosaic. If unable, default to basic (less pretty) pyplot method
        fig, ax = self.create_scene_capture_mosaic(scenario, 
                                                   format=format, 
                                                   fig_width=fig_width, 
                                                   base_resolution=base_resolution,
                                                   b_use_annotated_side_images=b_use_annotated_side_images,
                                                   other_vehicle_paths=other_vehicle_paths)
        
        b_mosaic = True
        if fig is None:
            b_mosaic = False
            fig, ax = plt.subplots(num=1) # Create new figure/axes
            main_ax = ax
            marker_scale = 1.
            fontsize = 12.
            # for border in main_ax.spines.values():
            #     border.set_linewidth(marker_scale)
        else:
            main_ax = ax["ortho_aerial"]
            for ax1 in ax.values():
                for border in ax1.spines.values():
                    border.set_linewidth(marker_scale)

        ssa_array = scenario.scene_description.ssa_array
        start_location = self.scenario_builder.opr_attr.get_default_start_location()
        goal_location = self.scenario_builder.opr_attr.get_default_goal_location()

        opt_path_x = scenario.opt_path[:,0].tolist()
        opt_path_y = scenario.opt_path[:,1].tolist()

        # Plot. Lower zorders get plotted first (closer to the background)
        main_ax.add_patch(plt.Circle(start_location, self.scenario_builder.start_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True, label="Obstacle-Free Zone"))
        main_ax.add_patch(plt.Circle(goal_location, self.scenario_builder.goal_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True))
        main_ax.add_patch(plt.Circle(goal_location, self.scenario_builder.opr_attr.goal_radius.default, linestyle='--', linewidth=1*marker_scale, color=self.COLOR_GOAL_LOCATION, fill=False, label="Goal Radius"))
        
        main_ax.scatter(start_location[0], start_location[1], s=marker_scale*(7**2), zorder=1, color=self.COLOR_START_LOCATION, marker='x', linewidths=1*marker_scale, label="Start Location")
        main_ax.scatter(goal_location[0], goal_location[1], s=marker_scale*(7**2), zorder=1, color=self.COLOR_GOAL_LOCATION, marker='x', linewidths=1*marker_scale, label="Goal Location")

        if not b_mosaic:
            # Plot visibile SSAs
            for ssa_type, ssa_type_info in self.SSA_PLOT_INFO.items():
                ssa_x = []
                ssa_y = []
                for i in range(len(self.scenario_builder.ssa_config)):
                    if self.scenario_builder.ssa_config[i].ssa_type == ssa_type:
                        layout = ssa_array[i]
                        for j in range(layout.num_instances):
                            if layout.visible[j]:
                                ssa_x.append(layout.x[j])
                                ssa_y.append(layout.y[j])

                if len(ssa_x) > 0:
                    main_ax.scatter(ssa_x, ssa_y, s=marker_scale*ssa_type_info["markersize"], zorder=2, color=ssa_type_info["color"], marker=ssa_type_info["marker"], linewidths=0, label=str.capitalize(ssa_type))

            main_ax.set_xlabel("X [m]", fontsize=fontsize)
            main_ax.set_ylabel("Y [m]", fontsize=fontsize)
            main_ax.xaxis.set_major_locator(ticker.MultipleLocator(self.scenario_builder.landscape_nominal_size))
            main_ax.yaxis.set_major_locator(ticker.MultipleLocator(self.scenario_builder.landscape_nominal_size))
            main_ax.set_aspect('equal', adjustable='box')
        else:
            main_ax.tick_params(axis="both", which="major", labelsize=fontsize, width=0.8*marker_scale, length=3.5*marker_scale, pad=3.5*marker_scale)

        main_ax.plot(opt_path_x, opt_path_y, zorder=3, linestyle='--', linewidth=1*marker_scale, color=self.COLOR_OPT_REF_PATH, label="Opt. Ref. Path")

        return fig, ax, b_mosaic
    
    def plot_scene_from_scenario(self, 
                                  scenario: Scenario, 
                                  file_name_prefix: str, 
                                  format : str = "portrait", 
                                  b_add_legend: bool = False, 
                                  b_use_annotated_side_images: bool = False,
                                  test_vehicle_paths: List[np.float32] = None):
        """Main function for plotting scenes.

        Args:
            - scenario: The scenaio object with the scene description and the base AV path
            - file_name_prefix: The base file path used for saving images. Do NOT include the extension (this will save both a .pdf and .png file)
            - format: Applies for creating the scene capture mosaic (defaul) with a large orthographic aerial view and two smaller images (front aerial and vehicle starting POV). 
                        If "landscape", then the smaller images are placed on the right side of the large orthographic view. If "portrait", then they are placed above it.
            - title: (Optional) Title for the figure. If None, then this function will create one based on the parameters provided.
            - b_use_annotated_side_images: Indicates if the scene capture should use the annotated side images.
            - test_vehicle_paths: (Optioanl) If we want to plot vehicle paths from the base AV and a secondary test AV, then provide those test AV paths here.
        """
        fig_width = 1.5 # Inches, change as needed
        base_resolution = 512

        font_ratio = 12/6.4
        ms_ratio = 1/6.4
        fontsize = fig_width * font_ratio
        marker_scale = fig_width * ms_ratio
        dpi = 500 * (1.5 / fig_width)

        if not self.b_saved_scene_legends:
            self.save_scene_legend(fontsize, marker_scale)
            self.b_saved_scene_legends = True

        fig, ax, b_mosaic = self.make_base_scene_plot(scenario, 
                                                      format=format, 
                                                      fig_width=fig_width,
                                                      fontsize=fontsize, 
                                                      marker_scale=marker_scale,
                                                      base_resolution=base_resolution,
                                                      b_use_annotated_side_images=b_use_annotated_side_images,
                                                      other_vehicle_paths=test_vehicle_paths)

        if b_mosaic:
            main_ax = ax["ortho_aerial"]
        else:
            main_ax = ax
            fontsize = 12
            marker_scale = 1.

        # Add base/main vehicle path
        b_labeled = False
        if test_vehicle_paths is None:
            vehicle_path_label = "Vehicle Path"
        else:
            vehicle_path_label = "Base AV Path"
        for vehicle_path in scenario.get_vehicle_paths():
            path_x = vehicle_path[:,0].tolist()
            path_y = vehicle_path[:,1].tolist()
            if not b_labeled:
                main_ax.plot(path_x, path_y, zorder=3, linewidth=1*marker_scale, color=self.COLOR_BASE_AV_PATH, alpha=0.5, label=vehicle_path_label)
                b_labeled = True
            else:
                main_ax.plot(path_x, path_y, zorder=3, linewidth=1*marker_scale, color=self.COLOR_BASE_AV_PATH, alpha=0.5)

        # Add test vehicle paths
        if test_vehicle_paths is not None:
            b_labeled = False
            for vehicle_path in test_vehicle_paths:
                path_x = vehicle_path[:,0].tolist()
                path_y = vehicle_path[:,1].tolist()
                if not b_labeled:
                    main_ax.plot(path_x, path_y, zorder=3, linewidth=1*marker_scale, color=self.COLOR_TEST_AV_PATH, alpha=0.5, label="Test AV Path")
                    b_labeled = True
                else:
                    main_ax.plot(path_x, path_y, zorder=3, linewidth=1*marker_scale, color=self.COLOR_TEST_AV_PATH, alpha=0.5)

        handles, labels = main_ax.get_legend_handles_labels()
        if not b_mosaic:
            if test_vehicle_paths is None:
                min_path_x, max_path_x, min_path_y, max_path_y = self.get_extreme_path_values(scenario.get_vehicle_paths())
            else:
                min_path_x, max_path_x, min_path_y, max_path_y = self.get_extreme_path_values(scenario.get_vehicle_paths() + test_vehicle_paths)

            xmin = 0. if 0. < min_path_x else min_path_x
            ymin = 0. if 0. < min_path_y else min_path_y
            xmax = self.scenario_builder.landscape_size[0] if max_path_x < self.scenario_builder.landscape_size[0] else max_path_x
            ymax = self.scenario_builder.landscape_size[1] if max_path_y < self.scenario_builder.landscape_size[1] else max_path_y
            main_ax.set_xlim([xmin - 5., xmax + 5.])
            main_ax.set_ylim([ymin - 5., ymax + 5.])

            fig.tight_layout(pad=0.5)
            fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, 1), fontsize=fontsize)
            fig.savefig(file_name_prefix + "_pyplot.pdf", bbox_inches="tight")
        else:
            fig.subplots_adjust(wspace=0., hspace=0.)
            if not b_add_legend:
                fig.savefig(file_name_prefix + f"_figwidth{fig_width}.pdf", bbox_inches="tight", pad_inches=0.1*marker_scale)
                fig.savefig(file_name_prefix + f"_figwidth{fig_width}.png", bbox_inches="tight", pad_inches=0.1*marker_scale, dpi=dpi)

                # Add legend
                if format == "landscape":
                    legend = fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, 0.05), fontsize=fontsize)
                elif format == "portrait":
                    legend = main_ax.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, -0.05), fontsize=fontsize)
                legend.get_frame().set_linewidth(0.8 * marker_scale)
                legend.get_frame().set_facecolor(self.COLOR_LEGEND)
                
                fig.savefig(file_name_prefix + f"_wlegend_figwidth{fig_width}.pdf", bbox_inches="tight", pad_inches=0.1*marker_scale)
                fig.savefig(file_name_prefix + f"_wlegend_figwidth{fig_width}.png", bbox_inches="tight", pad_inches=0.1*marker_scale, dpi=dpi)

        fig.clf()
        plt.close(fig)

    def _create_dummy_scene(self, marker_scale: float, b_test_path: bool = False):
        fig, ax = plt.subplots(num=1)

        ax.add_patch(plt.Circle((0.,0.), self.scenario_builder.start_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True, label="Obstacle-Free Zone"))
        ax.add_patch(plt.Circle((0.,0.), self.scenario_builder.opr_attr.goal_radius.default, linestyle='--', linewidth=1*marker_scale, color=self.COLOR_GOAL_LOCATION, fill=False, label="Goal Radius"))
        
        ax.scatter(0., 0., s=marker_scale*(7**2), zorder=1, color=self.COLOR_START_LOCATION, marker='x', linewidths=1*marker_scale, label="Start Location")
        ax.scatter(0., 0., s=marker_scale*(7**2), zorder=1, color=self.COLOR_GOAL_LOCATION, marker='x', linewidths=1*marker_scale, label="Goal Location")

        ax.plot([0.,1.], [0.,1.], zorder=3, linestyle='--', linewidth=1*marker_scale, color=self.COLOR_OPT_REF_PATH, label="Opt. Ref. Path")
        
        if not b_test_path:
            ax.plot([0.,1.], [0.,1.], zorder=3, linewidth=1*marker_scale, color=self.COLOR_BASE_AV_PATH, alpha=0.5, label="Vehicle Path")
        else:
            ax.plot([0.,1.], [0.,1.], zorder=3, linewidth=1*marker_scale, color=self.COLOR_BASE_AV_PATH, alpha=0.5, label="Base AV Path")
            ax.plot([0.,1.], [0.,1.], zorder=3, linewidth=1*marker_scale, color=self.COLOR_TEST_AV_PATH, alpha=0.5, label="Test AV Path")
        
        return fig, ax

    def save_scene_legend(self, fontsize: float, marker_scale: float):
        fig, ax = self._create_dummy_scene(marker_scale, False)
        ax.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=3, fontsize=fontsize)
        legend.get_frame().set_linewidth(0.8 * marker_scale)
        legend.get_frame().set_facecolor(self.COLOR_LEGEND)
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_3col.pdf"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_3col.png"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.clf()

        fig, ax = self._create_dummy_scene(marker_scale, False)
        ax.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=1, fontsize=fontsize)
        legend.get_frame().set_linewidth(0.8 * marker_scale)
        legend.get_frame().set_facecolor(self.COLOR_LEGEND)
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_1col.pdf"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_1col.png"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.clf()

        fig, ax = self._create_dummy_scene(marker_scale, False)
        ax.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=len(handles), fontsize=fontsize)
        legend.get_frame().set_linewidth(0.8 * marker_scale)
        legend.get_frame().set_facecolor(self.COLOR_LEGEND)
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_1row.pdf"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_1row.png"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.clf()

        fig, ax = self._create_dummy_scene(marker_scale, True)
        ax.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=3, fontsize=fontsize)
        legend.get_frame().set_linewidth(0.8 * marker_scale)
        legend.get_frame().set_facecolor(self.COLOR_LEGEND)
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_test_3col.pdf"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_test_3col.png"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.clf()

        fig, ax = self._create_dummy_scene(marker_scale, True)
        ax.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=1, fontsize=fontsize)
        legend.get_frame().set_linewidth(0.8 * marker_scale)
        legend.get_frame().set_facecolor(self.COLOR_LEGEND)
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_test_1col.pdf"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_test_1col.png"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.clf()
        
        fig, ax = self._create_dummy_scene(marker_scale, True)
        ax.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=len(handles), fontsize=fontsize)
        legend.get_frame().set_linewidth(0.8 * marker_scale)
        legend.get_frame().set_facecolor(self.COLOR_LEGEND)
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_test_1row.pdf"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.savefig(os.path.join(self.dirs["legends"], "scene_legend_base_test_1row.png"), bbox_inches="tight", pad_inches=0.1*marker_scale)
        fig.clf()

    def make_main_figure_plots(self):
        self.log("info", f"Making main figure plots...")
        fig_width = 1.5 # Change as needed

        font_ratio = 12/6.4
        ms_ratio = 1/6.4
        fontsize = fig_width * font_ratio
        marker_scale = fig_width * ms_ratio
        dpi = 500 * (1.5 / fig_width)
        
        id = 1432
        scenario = self.get_generated_scenario_from_id(id)
        start_location = self.scenario_builder.opr_attr.get_default_start_location()
        goal_location = self.scenario_builder.opr_attr.get_default_goal_location()
        opt_path_x = scenario.opt_path[:,0].tolist()
        opt_path_y = scenario.opt_path[:,1].tolist()

        ortho_aerial_image = cv2.cvtColor(cv2.imread(os.path.join(self.dirs["scene_captures"], f"scenario_{id}", f"ortho_aerial_pad0.png")), cv2.COLOR_BGR2RGB)
        ortho_aerial_image = cv2.resize(ortho_aerial_image, (512,512))

        # Save images in lower res
        ids = [21,420,1015,1432]
        for i in ids:
            img = cv2.imread(os.path.join(self.dirs["scene_captures"], f"scenario_{i}", f"perspective_front_aerial_45.png"))
            img = cv2.resize(img, (512,512))
            cv2.imwrite(os.path.join(self.dirs["figures"], f"main_bacre_figure_scenario_{i}.png"), img)

        # Fig 1
        fig, ax = plt.subplots()
        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_width)
        for border in ax.spines.values():
            border.set_linewidth(marker_scale)

        size = self.scenario_builder.landscape_nominal_size
        extent = (0., size, 0., size)
        ax.imshow(ortho_aerial_image, extent=extent, aspect="equal", interpolation="none")
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(size))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(size))
        ax.axis("off")
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[0], extent[1]])

        ax.add_patch(plt.Circle(start_location, self.scenario_builder.start_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True, label="Obstacle-Free Zone"))
        ax.add_patch(plt.Circle(goal_location, self.scenario_builder.goal_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True))
        ax.add_patch(plt.Circle(goal_location, self.scenario_builder.opr_attr.goal_radius.default, linestyle='--', linewidth=2*marker_scale, color=self.COLOR_GOAL_LOCATION, fill=False, label="Goal Radius"))
        ax.scatter(start_location[0], start_location[1], s=marker_scale*(7**2), zorder=1, color=self.COLOR_START_LOCATION, marker='x', linewidths=2*marker_scale, label="Start Location")
        ax.scatter(goal_location[0], goal_location[1], s=marker_scale*(7**2), zorder=1, color=self.COLOR_GOAL_LOCATION, marker='x', linewidths=2*marker_scale, label="Goal Location")
        ax.tick_params(axis="both", which="major", labelsize=fontsize, width=0.8*marker_scale, length=3.5*marker_scale, pad=3.5*marker_scale)
        ax.plot(opt_path_x, opt_path_y, zorder=3, linestyle='--', linewidth=2*marker_scale, color=self.COLOR_OPT_REF_PATH, label="Opt. Ref. Path")

        fig.savefig(os.path.join(self.dirs["figures"], "main_bacre_figure_ref_path.pdf"), bbox_inches="tight", pad_inches=0.)
        fig.savefig(os.path.join(self.dirs["figures"], "main_bacre_figure_ref_path.png"), bbox_inches="tight", pad_inches=0.)
        fig.clf()
        plt.close(fig)

        # Fig 2
        fig, ax = plt.subplots()
        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_width)
        for border in ax.spines.values():
            border.set_linewidth(marker_scale)

        size = self.scenario_builder.landscape_nominal_size
        extent = (0., size, 0., size)
        ax.imshow(ortho_aerial_image, extent=extent, aspect="equal", interpolation="none")
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(size))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(size))
        ax.axis("off")
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[0], extent[1]])

        ax.add_patch(plt.Circle(start_location, self.scenario_builder.start_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True, label="Obstacle-Free Zone"))
        ax.add_patch(plt.Circle(goal_location, self.scenario_builder.goal_obstacle_free_radius, linestyle='-', linewidth=0, color=self.COLOR_OBS_FREE_ZONE, alpha=0.15, fill=True))
        ax.add_patch(plt.Circle(goal_location, self.scenario_builder.opr_attr.goal_radius.default, linestyle='--', linewidth=2*marker_scale, color=self.COLOR_GOAL_LOCATION, fill=False, label="Goal Radius"))
        ax.scatter(start_location[0], start_location[1], s=marker_scale*(7**2), zorder=1, color=self.COLOR_START_LOCATION, marker='x', linewidths=2*marker_scale, label="Start Location")
        ax.scatter(goal_location[0], goal_location[1], s=marker_scale*(7**2), zorder=1, color=self.COLOR_GOAL_LOCATION, marker='x', linewidths=2*marker_scale, label="Goal Location")
        ax.tick_params(axis="both", which="major", labelsize=fontsize, width=0.8*marker_scale, length=3.5*marker_scale, pad=3.5*marker_scale)
        
        for vehicle_path in scenario.get_vehicle_paths():
            path_x = vehicle_path[:,0].tolist()
            path_y = vehicle_path[:,1].tolist()
            ax.plot(path_x, path_y, zorder=3, linewidth=2*marker_scale, color=self.COLOR_BASE_AV_PATH, alpha=0.5)

        fig.savefig(os.path.join(self.dirs["figures"], "main_bacre_figure_vehicle_path.pdf"), bbox_inches="tight", pad_inches=0.)
        fig.savefig(os.path.join(self.dirs["figures"], "main_bacre_figure_vehicle_path.png"), bbox_inches="tight", pad_inches=0.)
        fig.clf()
        plt.close(fig)

    def process_analyze_scenario_request(self, wid: int):
        """Process the AnalyzeScenario request from the specified worker
        
        Args:
            - wid: Worker ID

        Returns:
            True if no issues, False otherwise
        """
        worker : BACREWorkerRef = self.workers[wid]

        if worker.b_waiting_for_analyze_scenario_request or worker.analyze_scenario_request is None:
            return False
        
        # Save scene captures
        if len(worker.analyze_scenario_request.scene_captures) > 0:
            save_dir = os.path.join(self.dirs["scene_captures"], f"scenario_{worker.scenario.id}")
            os.makedirs(save_dir, exist_ok=True)
            for name, img_msg in zip(worker.analyze_scenario_request.scene_capture_names, worker.analyze_scenario_request.scene_captures):
                img = ros_img_converter.image_msg_to_numpy(img_msg, keep_as_3d_tensor=True)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_filepath = os.path.join(save_dir, name + ".png")
                cv2.imwrite(img_filepath, img)
        
        # Do not process rest of AnalyzeScenario request if the request only contained scene captures
        if worker.analyze_scenario_request.scene_capture_only:
            if len(worker.analyze_scenario_request.scene_captures) == 0:
                self.log("warn", "AnalyzeScenario request is scene capture only but 'scene_captures' field is empty.")
                return False
            else:
                self.log("info", f"Processed AnalyzeScenario request (scene capture only) for Worker {wid} / SN {worker.analyze_scenario_request.scenario_number}")
                return True

        if len(worker.analyze_scenario_request.vehicle_trajectory) == 0:
            self.log("warn", "AnalyzeScenario request field 'vehicle_trajectory' is empty. Requesting rerun.")
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return False
        
        # Get duration of trajectory
        start_stamp = worker.analyze_scenario_request.vehicle_trajectory[0].header.stamp # builtin_interfaces/msg/Time msg
        end_stamp = worker.analyze_scenario_request.vehicle_trajectory[-1].header.stamp
        sim_time = stamp_utils.get_stamp_dt(start_stamp, end_stamp)

        if worker.analyze_scenario_request.vehicle_sim_time > self.scenario_builder.opr_attr.sim_timeout_period.default + 0.25:
            self.log("warn", f"AnalyzeScenario request vehicle_sim_time of {worker.analyze_scenario_request.vehicle_sim_time} seconds is too long. Requesting rerun.")
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return False

        # Make sure time stamps are correct
        time_thresh = 1e-5
        if math.fabs(worker.analyze_scenario_request.vehicle_sim_time - sim_time) > time_thresh:
            raise Exception(f"Analyze scenario request for worker {wid}: Computed vehicle sim time of {sim_time:e} differs from expceted value of {worker.analyze_scenario_request.vehicle_sim_time:e} by more than {time_thresh:e} seconds.")

        # Extract full trajectory info into lists
        time_since_start = []
        xyz_full = []
        yaw_angles_full = []
        quat_full = []
        for odom in worker.analyze_scenario_request.vehicle_trajectory:
            time_since_start.append(stamp_utils.get_stamp_dt(start_stamp, odom.header.stamp))
            xyz_full.append(np.array([odom.pose.position.x, odom.pose.position.y, odom.pose.position.z], dtype=np.float32))
            
            quat = [odom.pose.orientation.w, odom.pose.orientation.x, odom.pose.orientation.y, odom.pose.orientation.z]
            euler = transforms3d.euler.quat2euler(quat, "rzyx")
            yaw_angles_full.append(euler[0])
            quat_full.append(quat)
        
        # Add scenario outcome
        outcome = ScenarioOutcome()
        outcome.sim_time = sim_time
        outcome.vehicle_path = np.array(xyz_full) # Each row is an (x,y,z) point
        outcome.yaw_angles = np.array(yaw_angles_full)
        outcome.vehicle_orientation = quat_full # Each element is a (w,x,y,z) quaternion
        worker.add_scenario_outcome(outcome) # Have to add outcome to scenario before we can compute the regret

        # Compute regret
        outcome.regret = self.compute_regret(wid)

        # worker.b_analyzed_scenario = True
        self.log("info", f"Processed AnalyzeScenario request for Worker {wid} / SN {worker.analyze_scenario_request.scenario_number}")
        return True

    def compute_regret(self, wid: int):
        """Compute the vehicle's regret for the scenario ran by the given worker
        
        Args:
            - wid: Worker ID
        """
        worker : BACREWorkerRef = self.workers[wid]
        regret = Regret()
        regret.set_to_zero()

        start_location = self.scenario_builder.opr_attr.get_default_start_location()
        goal_location = self.scenario_builder.opr_attr.get_default_goal_location()

        self.scenario_builder.configure_reference_agent(worker.run_scenario_request)

        opt_path_cost = worker.scenario.opt_path_cost
        opt_path_time = worker.scenario.opt_path_time
        
        vehicle_sim_time = worker.scenario.outcomes[-1].sim_time
        vehicle_path = worker.scenario.outcomes[-1].vehicle_path[:,0:2] # Each row is (x,y)
        vehicle_cost = self.scenario_builder.astar_path_planner.get_arbitrary_path_cost(vehicle_path)

        if worker.scenario.outcomes[-1].succeeded:
            # Need to adjust vehicle cost and sim time because the simulations ends the run when vehicle reaches the goal radius
            direction = goal_location - vehicle_path[-1] # Unnormalized direction from vehicle's last position to the goal point
            offsets = direction * np.linspace(0., 1., 100).reshape((-1,1)) # Rows are (deltaX, deltaY)
            remaining_path = vehicle_path[-1] + offsets # Each row is an (x,y) location, with 100 equally spaced points from the vehicle's last position to the goal point

            # Adjust cost and time
            adj_vehicle_cost = vehicle_cost + self.scenario_builder.astar_path_planner.get_arbitrary_path_cost(remaining_path)
            adj_vehicle_sim_time = vehicle_sim_time + (fast_norm(direction) / self.scenario_builder.nominal_vehicle_speed) # Assume vehicle travels at nominal speed (this is an underestimation, is this bad?)

            regret.cost_error = math.fabs(adj_vehicle_cost - opt_path_cost) / opt_path_cost
            regret.time_error = math.fabs(adj_vehicle_sim_time - opt_path_time) / opt_path_time
        else:
            regret.cost_error = math.fabs(vehicle_cost - opt_path_cost) / opt_path_cost
            regret.time_error = math.fabs(vehicle_sim_time - opt_path_time) / opt_path_time
            
            if self.b_regret_uses_equivalent_x_errors:
                normalizing_distance = fast_norm(goal_location - start_location)

                # Equivalent-cost position error
                if vehicle_cost >= opt_path_cost: 
                    # Optimal agent will have already reached the goal
                    regret.equiv_cost_pos_error = fast_norm(goal_location - vehicle_path[-1]) / normalizing_distance
                else:
                    # Find the location the optimal agent would reach if given a cost budget of vehicle_cost
                    equiv_cost_pos = cross_linear_interp(vehicle_cost, worker.scenario.opt_path_cost_values, worker.scenario.opt_path)
                    regret.equiv_cost_pos_error = fast_norm(equiv_cost_pos - vehicle_path[-1]) / normalizing_distance

                # Equivalent-time position error
                if vehicle_sim_time >= opt_path_time:
                    # Optimal agent will have already reached the goal
                    regret.equiv_time_pos_error = fast_norm(goal_location - vehicle_path[-1]) / normalizing_distance
                else:
                    # Find the location the optimal agent would reach if given vehicle_sim_time seconds
                    equiv_time_pos = cross_linear_interp(vehicle_sim_time, worker.scenario.opt_path_time_values, worker.scenario.opt_path)
                    regret.equiv_cost_pos_error = fast_norm(equiv_time_pos - vehicle_path[-1]) / normalizing_distance
        
        regret.value = regret.cost_error + regret.time_error + regret.equiv_cost_pos_error + regret.equiv_time_pos_error

        self.log("info", f"A* Reference: opt cost = {opt_path_cost:.4f}, opt time = {opt_path_time:.4f}")
        self.log("info", f"Vehicle (non-adjusted): path cost = {vehicle_cost:.4f}, sim time = {vehicle_sim_time:.4f}")
        self.log("info", f"Regret: cost error = {regret.cost_error:.4f}, time error = {regret.time_error:.4f}, eq cost pos error = {regret.equiv_cost_pos_error:.4f}, eq time pos error = {regret.equiv_time_pos_error:.4f}")

        return regret

    def num_workers_running_base_scenarios(self):
        n = 0
        for wid,worker in self.workers.items():
            worker : BACREWorkerRef
            n += worker.b_running_base_scenario
        return n

    def _perturb_str_attr(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        """Perturb a random structural attribute in the given scene description
        
        Args:
            - scenario_request: RunScenario request to modify

        Returns:
            True if successful, False otherwise
        """
        num_inactive, _ = self.scenario_builder.get_num_inactive_ssa(scenario_request.scene_description.ssa_array)
        num_active, _ = self.scenario_builder.get_num_active_ssa(scenario_request.scene_description.ssa_array)

        # Determine possible actions
        probs = []
        funcs = []
        if num_inactive > 0:
            probs.append(self.ssa_add_prob)
            funcs.append(self.scenario_builder.add_random_ssa)
        if num_active >= 2:
            probs.append(self.ssa_remove_prob)
            funcs.append(self.scenario_builder.remove_random_ssa)
        if num_active > 0: 
            probs.append(self.ssa_perturb_prob)
            funcs.append(self.scenario_builder.perturb_random_ssa)

        # Make sure array sums to 1
        probs = np.array(probs) / np.sum(probs)

        p = np.random.rand()
        for i in range(len(probs)):
            if p < np.sum(probs[:i+1]):
                funcs[i](scenario_request)
                return True

            # Instead of checking for the probability threshold, just run the function since it's the last thing we can run
            elif i == len(probs) - 1:
                funcs[i](scenario_request)
                return True

        return False # This return should almost never happen

    def can_create_new_scenario(self):
        """Indicates if we can create any more base scenarios"""
        if self.scenario_regret_manager.num_scenarios() < self.num_base_env:
            if self.scenario_regret_manager.num_scenarios() + self.num_workers_running_base_scenarios() < self.num_base_env:
                return True
            else:
                return False
        return True
    
    def get_parent_id_for_mutation(self):
        # Hyperparameter annealing
        n = self.scenario_regret_manager.num_scenarios() - self.num_base_env
        if n < self.anneal_duration:
            ratio = n/self.anneal_duration
            prob_rand = linear_interp(self.initial_prob_editing_random_env, self.prob_editing_random_env, ratio)
            rcr = linear_interp(self.initial_regret_curation_ratio, self.regret_curation_ratio, ratio)
        else:
            prob_rand = self.prob_editing_random_env
            rcr = self.regret_curation_ratio

        # Select a random parent scenario
        parent_id = None
        if np.random.rand() < prob_rand:
            parent_id = np.random.randint(self.scenario_regret_manager.num_scenarios())
        
        # Select parent based on primary selection method
        else:
            # Upper search quantile
            if self.parent_selection_method == self.PAR_SELECT_UPPER_QUANTILE:
                num_top_scenarios = math.ceil(self.scenario_regret_manager.num_scenarios() * self.upper_search_quantile)
                idx = np.random.randint(num_top_scenarios)
                parent_id = self.scenario_regret_manager.sorted_regrets[idx][0]
            
            # Regret curation ratio
            elif self.parent_selection_method == self.PAR_SELECT_RCR:
                largest_regret = self.scenario_regret_manager.sorted_regrets[0][1]
                smallest_regret = self.scenario_regret_manager.sorted_regrets[-1][1]
                cutoff_regret = largest_regret - rcr * (largest_regret - smallest_regret) # Want scenarios whose regrets lie in the set [cutoff_regret, largest_regret]
                
                if rcr == 1. or largest_regret == smallest_regret:
                    cutoff_num = self.scenario_regret_manager.num_scenarios()
                
                else:
                    cutoff_num = 0
                    for i in range(self.scenario_regret_manager.num_scenarios()):
                        if self.scenario_regret_manager.sorted_regrets[i][1] >= cutoff_regret:
                            cutoff_num += 1
                        else:
                            break
                
                # Pick from the best cutoff_num scenario IDs
                idx = np.random.randint(cutoff_num)
                parent_id = self.scenario_regret_manager.sorted_regrets[idx][0]
        
        return parent_id

    def add_reference_trajectory_to_scenario(self, scenario: Scenario, scenario_request):
        """
        Args:
            - scenario: Scenario to fill in reference trajectory data
            - scenario_request: The corresponding RunScenario request
        """
        # Add opt path, time, and cost to scenario
        opt_path, opt_path_len, opt_path_cost, opt_path_cost_values = self.scenario_builder.get_optimal_path(scenario_request)
        scenario.opt_path = np.array(opt_path)
        scenario.opt_path_len = opt_path_len
        scenario.opt_path_cost = opt_path_cost
        scenario.opt_path_cost_values = opt_path_cost_values
        scenario.opt_path_time = opt_path_len / self.scenario_builder.nominal_vehicle_speed

        # Get opt time values
        opt_path_time_values = [0.]
        for i in range(len(opt_path)-1):
            edge_length_ratio = fast_norm(opt_path[i+1] - opt_path[i]) / opt_path_len
            dt = edge_length_ratio * scenario.opt_path_time
            opt_path_time_values.append(opt_path_time_values[-1] + dt)
        opt_path_time_values[-1] = scenario.opt_path_time # Explicitly set this in case there are precision errors
        scenario.opt_path_time_values = opt_path_time_values

    def create_scenario(self, wid: int):
        """Create a new scenario for the given worker, reset/update any necessary variables, and submit the scenario request.
        
        Args:
            - wid: Worker ID
        """
        worker : BACREWorkerRef = self.workers[wid]
        self.scenario_builder : ScenarioBuilderAndRefAgent

        start_time = time.time()
        parent_id = None
        tree_level = 0

        if self.mode[self.MODE_BACRE]:
            if self.scenario_regret_manager.num_scenarios() < self.num_base_env:
                if self.scenario_regret_manager.num_scenarios() + self.num_workers_running_base_scenarios() < self.num_base_env:
                    # Create base scenario
                    self.log("info", "Creating base scenario...")
                    worker.b_running_base_scenario = True
                    scenario_request = self.scenario_builder.create_default_run_scenario_request()
                    self.scenario_builder.add_random_ssa(scenario_request)
                else:
                    return # Just have to wait until base population is created
            else:
                # Modify the parent scene description
                self.log("info", f"Creating child scenario with {self.num_mutations} mutations...")
                parent_id = self.get_parent_id_for_mutation()
                parent_scenario = self.get_generated_scenario_from_id(parent_id)
                scenario_request = self.scenario_builder.create_default_run_scenario_request()
                scenario_request.scene_description = copy.deepcopy(parent_scenario.scene_description)
                
                for _ in range(self.num_mutations):
                    if len(self.scenario_builder.ssa_attr.var_attrs) > 0 and len(self.scenario_builder.txt_attr.var_attrs) > 0:
                        if np.random.rand() > 0.5:
                            self._perturb_str_attr(scenario_request)
                        else:
                            self.scenario_builder.perturb_rand_txt_attr(scenario_request)
                    elif len(self.scenario_builder.ssa_attr.var_attrs) > 0:
                        self._perturb_str_attr(scenario_request)
                    else:
                        self.scenario_builder.perturb_rand_txt_attr(scenario_request)
        else:
            # Create random scenario with random number of SSAs
            scenario_request = self.scenario_builder.create_default_run_scenario_request()
            num_ssa = np.random.randint(self.scenario_builder.num_var_ssa_instances) + 1
            self.log("info", f"Creating random scenario with {num_ssa} SSAs...")
            for i in range(num_ssa):
                self.scenario_builder.add_random_ssa(scenario_request)

        # Do not do scene captures yet as the images can take up a LOT of space
        scenario_request.take_scene_capture = False
        scenario_request.scene_capture_only = False
        scenario_request.scene_capture_settings = copy.deepcopy(self.scene_capture_settings)

        worker.run_scenario_request = scenario_request        
        if parent_id is not None:
            tree_level = self.get_generated_scenario_from_id(parent_id).tree_level + 1
        worker.scenario = Scenario(self.scenario_regret_manager.get_new_temp_id(), parent_id, tree_level, scenario_request.scene_description)
        self.add_reference_trajectory_to_scenario(worker.scenario, scenario_request)
        worker.b_need_new_scenario = False

        self.log("info", f"Created new scenario (w/parent ID = {parent_id}) in {time.time() - start_time:.4f} seconds for worker {wid}")
        self.submit_run_scenario_request(wid)

    def load_vehicle_node_data(self, wid: int):
        """Load required vehicle node data into scenario outcome.
        
        Args:
            - wid: Worker ID

        Returns:
            True if successful, False otherwise.
        """
        worker : BACREWorkerRef = self.workers[wid]

        filepath = os.path.join(worker.vehicle_node_save_dir, "astar_path_snapshots.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                worker.scenario.outcomes[-1].astar_path_snapshots : List[AstarPathSnapshot] = pickle.load(f)
        else:
            self.log("error", f"Could not load file: {filepath}")
            return False

        filepath = os.path.join(worker.vehicle_node_save_dir, "astar_snapshots_summary.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                worker.scenario.outcomes[-1].astar_snapshots_summary : AstarSnapshotsSummary = pickle.load(f)
        else:
            self.log("error", f"Could not load file: {filepath}")
            return False

        filepath = os.path.join(worker.vehicle_node_save_dir, "path_follower_snapshots.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                worker.scenario.outcomes[-1].path_follower_snapshots : List[PathFollowerTrackingSnapshot] = pickle.load(f)
        else:
            self.log("error", f"Could not load file: {filepath}")
            return False
        
        filepath = os.path.join(worker.vehicle_node_save_dir, "path_follower_snapshots_summary.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                worker.scenario.outcomes[-1].path_follower_snapshots_summary : PathFollowerSnapshotsSummary = pickle.load(f)
        else:
            self.log("error", f"Could not load file: {filepath}")
            return False
        
        return True

    def run_bacre_step(self, wid: int):
        """Run a BACRE step for the given worker

        Args:
            - wid: Worker ID
        """
        if not self.mode[self.MODE_BACRE]:
            return
        
        worker : BACREWorkerRef = self.workers[wid]

        if not self.are_all_vehicle_nodes_ready(wid):
            return
        
        if wid not in self.worker_mspids.keys():
            self.worker_mspids[wid] = 1 # This value is meaningless, it's just so we can add the worker to the worker_mspids dict

        worker.vehicle_node_save_dir = self.temp_vehicle_node_save_dirs[wid]
        worker.b_save_minimal = True

        if worker.b_need_new_scenario:
            if self.can_create_new_scenario():
                self.log("info", "-" * 30 + f'BACRE STEP (wid {wid})' + "-" * 30, b_log_ros=False)
            self.create_scenario(wid)
            return
                
        # Block until ready to analyze scenario
        if worker.b_waiting_for_analyze_scenario_request or worker.analyze_scenario_request is None:
            return

        self.log("info", "-" * 30 + f'BACRE STEP (wid {wid})' + "-" * 30, b_log_ros=False)
        if not self.process_analyze_scenario_request(wid):
            return
        
        succ_fail = "Success" if worker.scenario.outcomes[-1].succeeded else "Fail"
        self.log("info", f"Worker {wid} / SN {worker.scenario_number} / Temp ID {worker.scenario.temp_id} / Run {worker.num_scenario_runs}: {succ_fail}, Regret = {worker.scenario.outcomes[-1].regret.value:.4f}")
        
        # Add vehicle node data to scenario outcome
        b_loaded_vehicle_node_data = self.load_vehicle_node_data(wid)

        # Recreate vehicle node save dir
        if os.path.isdir(worker.vehicle_node_save_dir):
            shutil.rmtree(worker.vehicle_node_save_dir)
        os.makedirs(worker.vehicle_node_save_dir, exist_ok = True)
        
        if not b_loaded_vehicle_node_data:
            self.log("error", f"At least one vehicle node file was not loaded or does not exist. Will remove last scenario outcome and resubmit RunScenario request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        # Verify the simulation did not miss too many control commands. Use tight tolerance of 2%.
        num_vehicle_control_messages_sent = worker.scenario.outcomes[-1].path_follower_snapshots_summary.num_control_messages
        if math.fabs(worker.analyze_scenario_request.num_vehicle_control_messages - num_vehicle_control_messages_sent) > round(0.02*num_vehicle_control_messages_sent):
            self.log("error", f"UE4 simulation missed too many control commands. Sent {num_vehicle_control_messages_sent} but only {worker.analyze_scenario_request.num_vehicle_control_messages} were received. Will try again.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
        
        # Check for rerun request
        if worker.b_vehicle_node_requested_rerun and worker.num_rerun_requests <= self.MAX_RERUN_REQUESTS:
            self.log("warn", f"Resubmitting RunScenario request due to rerun request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        # Run required number of scenario runs
        if worker.num_scenario_runs < self.num_runs_per_scenario:
            worker.run_scenario_request.take_scene_capture = False
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
    
        worker.scenario.estimate_regret()
        self.scenario_regret_manager.add_scenario(worker.scenario)
        self.log("info", f"Worker {wid} / SN {worker.scenario_number} / Temp ID {worker.scenario.temp_id} / Avg. Regret = {worker.scenario.avg_regret.value:.4f}")
        self.log("info", f"Num. Scenarios Collected: {self.scenario_regret_manager.num_scenarios()}")
        self.generated_scenarios[worker.scenario.id] = worker.scenario
        self.save_generated_scenario(worker.scenario)

        if self.scenario_regret_manager.num_scenarios() <= self.num_base_env:
            self.save_scenario_regret_manager()

        if worker.scenario.avg_regret.value > self.max_regret_achieved:
            self.max_regret_achieved = worker.scenario.avg_regret.value
            self.plot_max_regret_scene()

        if self.scenario_regret_manager.num_scenarios() % self.data_save_freq == 0:
            self.save_scenario_regret_manager()
            file_name = os.path.join(self.dirs['figures'], 'last_generated_scene')
            self.plot_scene_from_scenario(worker.scenario, file_name)
            self.plot_scenario_rankings()

            info_file_name = os.path.join(self.dirs['figures'], 'last_generated_scene_info.txt')
            with open(info_file_name, "w") as f:
                f.write(f"Iter {self.scenario_regret_manager.num_scenarios()} / Avg. Regret: {worker.scenario.avg_regret.value:.2f}")

        worker.reset()

        # Check if done creating scenarios
        if self.scenario_regret_manager.num_scenarios() >= self.num_env_to_generate:
            self.log("info", f"Done creating adversarial scenes. Created {self.num_env_to_generate} scenes. Entering replay mode.")
            self.mode[self.MODE_BACRE] = False
            self.mode[self.MODE_SCENE_CAPTURE] = True

            # Cancel any attempts to send non-local WIDs their latest RunScenario requests (if they were previously unsuccessful)
            for wid2 in self.recognized_wids:
                self.cancel_submitting_run_scenario_request(wid2)

            self.save_scenario_regret_manager()
            self.plot_scenario_rankings()
            self.analyze_generated_scenarios_snapshots()
            self.process_scene_capture_request()

            return

    def analyze_replay_scenario(self, scenario_dir: str, scenario: Scenario = None):
        """Analyze a replay scenario
        
        Args:
            - scenario_dir: The directory containing the replay scenario
            - scenario: The replayed scenario. If None, it will get loaded from the above directory (if unable to load, then the function will return)
        """
        replay_scenario = scenario
        if scenario is None:
            file_path = os.path.join(scenario_dir, f"scenario.pkl")
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    replay_scenario = pickle.load(f)
            else:
                return
        
        original_scenario = self.get_generated_scenario_from_id(replay_scenario.id)
        
        # Plot replay scene summary
        file_name = os.path.join(scenario_dir, 'replay_scene_summary')
        self.plot_scene_from_scenario(replay_scenario, file_name)

        # Plot original scene summary
        scene_filename = os.path.join(scenario_dir, "original_scene_summary")
        self.plot_scene_from_scenario(original_scenario, scene_filename)

        file_name = os.path.join(scenario_dir, "orig_scene_with_replay")
        self.plot_scene_from_scenario(original_scenario, file_name, test_vehicle_paths=replay_scenario.get_vehicle_paths())

        if self.scenario_analysis is not None:
            # Analyze snapshots and write values to txt file
            (mean_astar_replan_angle, 
                max_astar_replan_angle, 
                mean_path_follower_angle, 
                max_path_follower_angle, 
                avg_loops_per_path,
                weighted_obstacle_detection_distance,
                mean_image_processing_durations, 
                max_image_processing_durations, 
                mean_astar_replan_iterations, 
                max_astar_replan_iterations, 
                mean_astar_replan_durations, 
                max_astar_replan_durations) = self.analyze_scenario_snapshots(replay_scenario)
            
            # Write stats to txt file
            analysis_results_filepath = os.path.join(scenario_dir, "scenario_analysis.txt")
            id = replay_scenario.id
            with open(analysis_results_filepath, "w") as f:
                f.write(f"Scenario Analysis and Comparsion:\n")

                f.write(f"Original Avg. Regret: {original_scenario.avg_regret.value:.4f} \n")
                f.write(f"Replay Avg. Regret: {replay_scenario.avg_regret.value:.4f} \n")

                f.write(f"Mean A* Replan Angle [deg]: Orig {self.scenario_analysis.mean_astar_replan_angle[id]:.4f}, New {mean_astar_replan_angle:.4f} \n")
                f.write(f"Max A* Replan Angle [deg]: Orig {self.scenario_analysis.max_astar_replan_angle[id]:.4f}, New {max_astar_replan_angle:.4f} \n")

                f.write(f"Mean Path Follower Angle [deg]: Orig {self.scenario_analysis.mean_path_follower_angle[id]:.4f}, New {mean_path_follower_angle:.4f} \n")
                f.write(f"Max Path Follower Angle [deg]: Orig {self.scenario_analysis.max_path_follower_angle[id]:.4f}, New {max_path_follower_angle:.4f} \n")

                f.write(f"Avg. Loops Per Path: Orig {self.scenario_analysis.avg_loops_per_path[id]:.4f}, New {avg_loops_per_path:.4f} \n")
                f.write(f"Weighted Obstacle Detection Distance [m]: Orig {self.scenario_analysis.weighted_obstacle_detection_distance[id]:.4f}, New {weighted_obstacle_detection_distance:.4f} \n")

                f.write(f"Mean Image Processing Time [s]: Orig {self.scenario_analysis.mean_image_processing_durations[id]:.4f}, New {mean_image_processing_durations:.4f} \n")
                f.write(f"Max Image Processing Time [s]: Orig {self.scenario_analysis.max_image_processing_durations[id]:.4f}, New {max_image_processing_durations:.4f} \n")

                f.write(f"Mean A* Replan Iterations: Orig {self.scenario_analysis.mean_astar_replan_iterations[id]:.4f}, New {mean_astar_replan_iterations:.4f} \n")
                f.write(f"Max A* Replan Iterations: Orig {self.scenario_analysis.max_astar_replan_iterations[id]:.4f}, New {max_astar_replan_iterations:.4f} \n")

                f.write(f"Mean A* Replan Time [s]: Orig {self.scenario_analysis.mean_astar_replan_durations[id]:.4f}, New {mean_astar_replan_durations:.4f} \n")
                f.write(f"Max A* Replan Time [s]: Orig {self.scenario_analysis.max_astar_replan_durations[id]:.4f}, New {max_astar_replan_durations:.4f} \n")

        else:
            self.log("warn", f"ScenarioAnalysis is None. Made plots, but cannot finish analyzing replay scenario.")

    def create_replay_worker_save_dir(self, wid: int):
        """Create the save directory for this replay worker's ROS nodes. If replaying from a different experiment, that relative path will appear in the save path.
        
        Args:
            - wid: Worker ID
        """
        worker : BACREWorkerRef = self.workers[wid]
        replay_num = worker.num_scenario_runs + 1

        worker.b_save_minimal = False
        rank = self.replay_rankings[self.worker_mspids[wid]]
        worker.vehicle_node_save_dir = os.path.join(self.dirs['replay'], f"rankings", f"rank_{rank}", f"replay_{replay_num}")
        
        if replay_num == 1:
            parent_dir = os.path.abspath(os.path.join(worker.vehicle_node_save_dir, '..'))
            if os.path.isdir(parent_dir):
                shutil.rmtree(parent_dir)

        # Recreate temp vehicle node save dir
        if os.path.isdir(self.temp_vehicle_node_save_dirs[wid]):
            shutil.rmtree(self.temp_vehicle_node_save_dirs[wid])
        os.makedirs(self.temp_vehicle_node_save_dirs[wid], exist_ok = True)

        # Recreate replay vehicle save dir
        if os.path.isdir(worker.vehicle_node_save_dir):
            shutil.rmtree(worker.vehicle_node_save_dir)
        os.makedirs(worker.vehicle_node_save_dir, exist_ok = True)

    def replay_scenario(self, wid: int, scenario: Scenario):
        """Replay a scenario for a given worker
        
        Args:
            - wid: Worker ID
            - scenario: The scenario to be replayed (argument will not be modified)
        """
        worker : BACREWorkerRef = self.workers[wid]
        worker.reset()

        worker.scenario = copy.deepcopy(scenario)
        worker.scenario.outcomes.clear()
        worker.run_scenario_request = self.scenario_builder.create_default_run_scenario_request()
        worker.run_scenario_request.scene_description = worker.scenario.scene_description

        self.log("info", f"Replaying scenario for Worker {wid}")
        self.submit_run_scenario_request(wid)

    def run_replay_step(self, wid: int):
        """Run a replay step for the given worker
        
        Args:
            - wid: Worker ID
        """
        worker : BACREWorkerRef = self.workers[wid]

        if not self.mode[self.MODE_REPLAY]:
            return

        if not self.are_all_vehicle_nodes_ready(wid):
            return
            
        if wid not in self.worker_mspids.keys():
            if self.num_mspids_remaining() > 0 and not worker.b_waiting_for_analyze_scenario_request:
                self.log("info", "-" * 30 + f"REPLAY (wid {wid})" + "-" * 30, b_log_ros=False)
                self.worker_mspids[wid] = self.get_next_mspid()
                scenario_idx = self.worker_mspids[wid]
                self.replay_scenario(wid, self.replay_scenarios[scenario_idx])
                self.create_replay_worker_save_dir(wid)
                return
            else:
                return
        
        # Block until ready to analyze scenario
        if worker.b_waiting_for_analyze_scenario_request or worker.analyze_scenario_request is None:
            return

        self.log("info", "-" * 30 + f"REPLAY (wid {wid})" + "-" * 30, b_log_ros=False)
        if not self.process_analyze_scenario_request(wid):
            return

        self.log("info", f"Replay Worker {wid} / Scenario Index {self.worker_mspids[wid]} / Replay {worker.num_scenario_runs}: Regret = {worker.scenario.outcomes[-1].regret.value:.4f}")

        # Add vehicle node data to scenario outcome
        b_loaded_vehicle_node_data = self.load_vehicle_node_data(wid)
        
        if not b_loaded_vehicle_node_data:
            self.log("error", f"At least one vehicle node file was not loaded or does not exist. Will remove last scenario outcome and resubmit RunScenario request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
        
        # Verify the simulation did not miss too many control commands. Use tight tolerance of 2%.
        num_vehicle_control_messages_sent = worker.scenario.outcomes[-1].path_follower_snapshots_summary.num_control_messages
        if math.fabs(worker.analyze_scenario_request.num_vehicle_control_messages - num_vehicle_control_messages_sent) > round(0.02*num_vehicle_control_messages_sent):
            self.log("error", f"UE4 simulation missed too many control commands. Sent {num_vehicle_control_messages_sent} but only {worker.analyze_scenario_request.num_vehicle_control_messages} were received. Will try again.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
        
        # Check for rerun request
        if worker.b_vehicle_node_requested_rerun and worker.num_rerun_requests <= self.MAX_RERUN_REQUESTS:
            self.log("warn", f"Resubmitting RunScenario request due to rerun request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        # Replay same scenario again, or advance to another scenario to replay
        if worker.num_scenario_runs < self.num_replays_per_scenario:
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            self.create_replay_worker_save_dir(wid)
        else:
            parent_dir = os.path.abspath(os.path.join(worker.vehicle_node_save_dir, '..'))
            
            worker.scenario.estimate_regret()
            self.log("info", f"Replay Worker {wid} / Scenario Index {self.worker_mspids[wid]} / Replay {worker.num_scenario_runs}: Avg. Regret = {worker.scenario.avg_regret.value:.4f}")
            
            # Save replay scenario with all outcomes
            replay_scenario_file = os.path.join(parent_dir, "scenario.pkl")
            with open(replay_scenario_file, 'wb') as f:
                pickle.dump(worker.scenario, f, pickle.HIGHEST_PROTOCOL)
            self.log("info", f"Saved replay scenario to file: {replay_scenario_file}")

            self.analyze_replay_scenario(parent_dir, worker.scenario)
            
            self.mspids_completed.append(self.worker_mspids[wid])
            self.worker_mspids.pop(wid)

            if self.num_mspids_remaining() > 0:
                self.worker_mspids[wid] = self.get_next_mspid()
                scenario_idx = self.worker_mspids[wid]
                self.replay_scenario(wid, self.replay_scenarios[scenario_idx])
                self.create_replay_worker_save_dir(wid)
            else:
                self.log("info", f"Worker {wid} done with replay")

            if len(self.mspids_completed) == len(self.mspids):
                self.mode[self.MODE_REPLAY] = False
                self.log("info", f"Finished replay mode")

                return

    def run_random_step(self, wid: int):
        """Run a random step for the given worker
        
        Args:
            - wid: Worker ID
        """
        if not self.mode[self.MODE_RANDOM]:
            return

        if not self.are_all_vehicle_nodes_ready(wid):
            return

        worker : BACREWorkerRef = self.workers[wid]
        worker.vehicle_node_save_dir = self.temp_vehicle_node_save_dirs[wid]
        worker.b_save_minimal = True

        if wid not in self.worker_mspids.keys():
            self.worker_mspids[wid] = 1 # This value is meaningless, it's just so we can add the worker to the worker_mspids dict

        if worker.b_need_new_scenario:
            self.log("info", "-" * 30 + f"RANDOM STEP (wid {wid})" + "-" * 30, b_log_ros=False)
            self.create_scenario(wid)
            return
                
        # Block until ready to analyze scenario
        if worker.b_waiting_for_analyze_scenario_request or worker.analyze_scenario_request is None:
            return

        self.log("info", "-" * 30 + f"RANDOM STEP (wid {wid})" + "-" * 30, b_log_ros=False)
        if not self.process_analyze_scenario_request(wid):
            return

        succ_fail = "Success" if worker.scenario.outcomes[-1].succeeded else "Fail"
        self.log("info", f"Worker {wid} / SN {worker.scenario_number} / Temp ID {worker.scenario.temp_id} / Run {worker.num_scenario_runs}: {succ_fail}, Regret = {worker.scenario.outcomes[-1].regret.value:.4f}")
        
        # Add vehicle node data to scenario outcome
        b_loaded_vehicle_node_data = self.load_vehicle_node_data(wid)
        
        # Recreate vehicle node save dir
        if os.path.isdir(worker.vehicle_node_save_dir):
            shutil.rmtree(worker.vehicle_node_save_dir)
        os.makedirs(worker.vehicle_node_save_dir, exist_ok = True)

        if not b_loaded_vehicle_node_data:
            self.log("error", f"At least one vehicle node file was not loaded or does not exist. Will remove last scenario outcome and resubmit RunScenario request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
        
        # Verify the simulation did not miss too many control commands. Use tight tolerance of 2%.
        num_vehicle_control_messages_sent = worker.scenario.outcomes[-1].path_follower_snapshots_summary.num_control_messages
        if math.fabs(worker.analyze_scenario_request.num_vehicle_control_messages - num_vehicle_control_messages_sent) > round(0.02*num_vehicle_control_messages_sent):
            self.log("error", f"UE4 simulation missed too many control commands. Sent {num_vehicle_control_messages_sent} but only {worker.analyze_scenario_request.num_vehicle_control_messages} were received. Will try again.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
        
        # Check for rerun request
        if worker.b_vehicle_node_requested_rerun and worker.num_rerun_requests <= self.MAX_RERUN_REQUESTS:
            self.log("warn", f"Resubmitting RunScenario request due to rerun request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        # Run required number of scenario runs
        if worker.num_scenario_runs < self.num_runs_per_scenario:
            worker.run_scenario_request.take_scene_capture = False
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        worker.scenario.estimate_regret()
        self.scenario_regret_manager.add_scenario(worker.scenario)
        self.log("info", f"Worker {wid} / SN {worker.scenario_number} / Temp ID {worker.scenario.temp_id} / Avg. Regret = {worker.scenario.avg_regret.value:.4f}")
        self.log("info", f"Num. Scenarios Collected: {self.scenario_regret_manager.num_scenarios()}")
        self.generated_scenarios[worker.scenario.id] = worker.scenario
        self.save_generated_scenario(worker.scenario)

        if self.scenario_regret_manager.num_scenarios() <= self.num_base_env:
            self.save_scenario_regret_manager()

        if worker.scenario.avg_regret.value > self.max_regret_achieved:
            self.max_regret_achieved = worker.scenario.avg_regret.value
            self.plot_max_regret_scene()

        if self.scenario_regret_manager.num_scenarios() % self.data_save_freq == 0:
            self.save_scenario_regret_manager()
            file_name = os.path.join(self.dirs['figures'], 'last_generated_scene')
            self.plot_scene_from_scenario(worker.scenario, file_name)
            self.plot_scenario_rankings()

            info_file_name = os.path.join(self.dirs['figures'], 'last_generated_scene_info.txt')
            with open(info_file_name, "w") as f:
                f.write(f"Iter {self.scenario_regret_manager.num_scenarios()} / Avg. Regret: {worker.scenario.avg_regret.value:.2f}")

        worker.reset()

        # Check if done creating scenarios
        if self.scenario_regret_manager.num_scenarios() >= self.num_env_to_generate:
            self.log("info", f"Done creating random scenes. Created {self.num_env_to_generate} scenes. Entering replay mode.")
            self.mode[self.MODE_RANDOM] = False
            self.mode[self.MODE_SCENE_CAPTURE] = True

            # Cancel any attempts to send non-local WIDs their latest RunScenario requests (if they were previously unsuccessful)
            for wid2 in self.recognized_wids:
                self.cancel_submitting_run_scenario_request(wid2)

            self.save_scenario_regret_manager()
            self.plot_scenario_rankings()
            self.analyze_generated_scenarios_snapshots()
            self.process_scene_capture_request()

            return

    def run_scenario_for_cga(self, wid: int, temp_id: int):
        worker : BACREWorkerRef = self.workers[wid]
        
        worker.scenario = self.load_cga_temp_scenario(temp_id)
        worker.scenario.outcomes.clear()
        worker.run_scenario_request = self.scenario_builder.create_default_run_scenario_request()
        worker.run_scenario_request.scene_description = worker.scenario.scene_description

        worker.run_scenario_request.take_scene_capture = False
        worker.run_scenario_request.scene_capture_only = False
        worker.run_scenario_request.scene_capture_settings = copy.deepcopy(self.scene_capture_settings)

        self.log("info", f"Running CGA scenario for Worker {wid} / Temp ID {temp_id}")
        self.submit_run_scenario_request(wid)

    def run_cga_step(self, wid: int):
        """Run a CGA step for the given worker

        Args:
            - wid: Worker ID
        """
        if not self.mode[self.MODE_CGA]:
            return
        
        worker : BACREWorkerRef = self.workers[wid]

        # Create scenario requests for a new population
        if len(self.mspids) == 0 and self.cga_population_manager.num_populations() < self.cga_num_populations:
            start_time = time.time()

            # Create parent IDs and temp IDs for new population
            if self.cga_population_manager.num_populations() == 0:
                cga_scenario_temp_ids = np.arange(self.cga_population_size)
            else:
                cga_scenario_temp_ids = np.arange(self.cga_population_size - self.cga_num_families_per_population*2)
                for i in range(self.cga_num_families_per_population):
                    p1_id = self.scenario_regret_manager.sorted_regrets[i*2][0]
                    p2_id = self.scenario_regret_manager.sorted_regrets[i*2 + 1][0]
                    self.cga_parent_ids.append((p1_id, p2_id))
                    
                    self.cga_population_idxs.append(p1_id)
                    self.cga_population_idxs.append(p2_id)
            cga_scenario_temp_ids += self.scenario_regret_manager.num_scenarios()
            self.mspids = cga_scenario_temp_ids.tolist()

            # Check if scenario requests were already created
            b_precompute_population_scenarios = False
            for temp_id in self.mspids:
                file_path = os.path.join(self.cga_temp_save_dir, f"scenario_temp_{temp_id}.pkl")
                if not os.path.exists(file_path):
                    b_precompute_population_scenarios = True
                    break

            if b_precompute_population_scenarios:
                chunk_size = math.ceil(len(self.mspids) / self.cga_num_families_per_population)
                processes : List[multiprocessing.Process] = []
                pid = 0
                for i in range(0, len(self.mspids), chunk_size):
                    if len(self.cga_parent_ids) > 0:
                        scenario1 = self.get_generated_scenario_from_id(self.cga_parent_ids[pid][0])
                        scenario2 = self.get_generated_scenario_from_id(self.cga_parent_ids[pid][1])
                    else:
                        scenario1 = None
                        scenario2 = None

                    p = multiprocessing.Process(target = multiprocess_create_cga_scenarios, 
                                                args = (pid, 
                                                        copy.deepcopy(scenario1),
                                                        copy.deepcopy(scenario2),
                                                        self.cga_mutation_prob,
                                                        copy.deepcopy(self.scenario_builder),
                                                        self.mspids[i:i+chunk_size],
                                                        self.cga_temp_save_dir
                                                        )
                                                )
                    p.start()
                    processes.append(p)
                    pid += 1
                for p in processes:
                    p.join()

                wall_time = (time.time() - start_time)/60.
                self.log("info", f"Finished computing reference trajectories for new CGA population in {wall_time:.2} minutes")
            else:
                self.log("info", f"CGA population scenario requests already created.")
        
        if not self.are_all_vehicle_nodes_ready(wid):
            return

        worker.vehicle_node_save_dir = self.temp_vehicle_node_save_dirs[wid]
        worker.b_save_minimal = True

        if wid not in self.worker_mspids.keys():
            if self.num_mspids_remaining() > 0 and not worker.b_waiting_for_analyze_scenario_request:
                self.log("info", "-" * 30 + f"CGA STEP (wid {wid})" + "-" * 30, b_log_ros=False)
                self.worker_mspids[wid] = self.get_next_mspid()
                self.run_scenario_for_cga(wid, self.worker_mspids[wid])
                return
            else:
                return
                
        # Block until ready to analyze scenario
        if worker.b_waiting_for_analyze_scenario_request or worker.analyze_scenario_request is None:
            return
        
        self.log("info", "-" * 30 + f"CGA STEP (wid {wid})" + "-" * 30, b_log_ros=False)
        if not self.process_analyze_scenario_request(wid):
            return
        
        succ_fail = "Success" if worker.scenario.outcomes[-1].succeeded else "Fail"
        self.log("info", f"Worker {wid} / SN {worker.scenario_number} / Temp ID {worker.scenario.temp_id} / Run {worker.num_scenario_runs}: {succ_fail}, Regret = {worker.scenario.outcomes[-1].regret.value:.4f}")
        
        # Add vehicle node data to scenario outcome
        b_loaded_vehicle_node_data = self.load_vehicle_node_data(wid)

        # Recreate vehicle node save dir
        if os.path.isdir(worker.vehicle_node_save_dir):
            shutil.rmtree(worker.vehicle_node_save_dir)
        os.makedirs(worker.vehicle_node_save_dir, exist_ok = True)
        
        if not b_loaded_vehicle_node_data:
            self.log("error", f"At least one vehicle node file was not loaded or does not exist. Will remove last scenario outcome and resubmit RunScenario request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        # Verify the simulation did not miss too many control commands. Use tight tolerance of 2%.
        num_vehicle_control_messages_sent = worker.scenario.outcomes[-1].path_follower_snapshots_summary.num_control_messages
        if math.fabs(worker.analyze_scenario_request.num_vehicle_control_messages - num_vehicle_control_messages_sent) > round(0.02*num_vehicle_control_messages_sent):
            self.log("error", f"UE4 simulation missed too many control commands. Sent {num_vehicle_control_messages_sent} but only {worker.analyze_scenario_request.num_vehicle_control_messages} were received. Will try again.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
        
        # Check for rerun request
        if worker.b_vehicle_node_requested_rerun and worker.num_rerun_requests <= self.MAX_RERUN_REQUESTS:
            self.log("warn", f"Resubmitting RunScenario request due to rerun request.")
            worker.remove_last_scenario_outcome()
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return

        # Run required number of scenario runs
        if worker.num_scenario_runs < self.num_runs_per_scenario:
            worker.run_scenario_request.take_scene_capture = False
            self.submit_run_scenario_request(wid, b_reset_num_rerun_requests=False)
            return
    
        worker.scenario.estimate_regret()
        self.scenario_regret_manager.add_scenario(worker.scenario)

        self.mspids_completed.append(worker.scenario.temp_id)
        self.cga_population_idxs.append(worker.scenario.id)

        self.log("info", f"Worker {wid} / SN {worker.scenario_number} / Temp ID {worker.scenario.temp_id} / Avg. Regret = {worker.scenario.avg_regret.value:.4f}")
        self.log("info", f"Total Scenarios Collected: {self.scenario_regret_manager.num_scenarios()}")
        self.log("info", f"Num. Scenarios Collected in current CGA Population: {len(self.mspids_completed)}")
        self.generated_scenarios[worker.scenario.id] = worker.scenario
        self.save_generated_scenario(worker.scenario)

        if worker.scenario.avg_regret.value > self.max_regret_achieved:
            self.max_regret_achieved = worker.scenario.avg_regret.value
            self.plot_max_regret_scene()

        worker.reset()
        self.worker_mspids.pop(wid)

        # Check if population is done
        if self.num_mspids_remaining() > 0:
            self.worker_mspids[wid] = self.get_next_mspid()
            self.run_scenario_for_cga(wid, self.worker_mspids[wid])
            return
        
        if len(self.mspids_completed) == len(self.mspids):
            self.cga_population_manager.population_idxs.append(self.cga_population_idxs)
            self.cga_population_manager.population_parent_idxs.append(self.cga_parent_ids)
            self.cga_population_manager.max_population_regret.append(self.max_regret_achieved)

            regret_sum = 0.
            for id in self.cga_population_idxs:
                regret_sum += self.get_generated_scenario_from_id(id).avg_regret.value
            self.cga_population_manager.avg_population_regret.append(regret_sum / self.cga_population_size)
            
            self.mspids.clear()
            self.mspids_completed.clear()
            self.cga_population_idxs = []
            self.cga_parent_ids = []

            self.save_scenario_regret_manager()
            self.save_cga_population_manager()

            self.plot_scenario_rankings()

            # Recreate temp vehicle node save dir
            if os.path.isdir(self.cga_temp_save_dir):
                shutil.rmtree(self.cga_temp_save_dir)
            os.makedirs(self.cga_temp_save_dir, exist_ok = True)
            self.log("info", f"Emptied CGA temp save dir")

        # Check if done creating scenarios
        if self.cga_population_manager.num_populations() >= self.cga_num_populations:
            self.log("info", f"Done creating CGA populations. Created {self.cga_population_manager.num_populations()} populations. Entering replay mode.")
            self.mode[self.MODE_CGA] = False
            self.mode[self.MODE_SCENE_CAPTURE] = True

            # Cancel any attempts to send non-local WIDs their latest RunScenario requests (if they were previously unsuccessful)
            for wid2 in self.recognized_wids:
                self.cancel_submitting_run_scenario_request(wid2)

            self.plot_scenario_rankings()
            self.analyze_generated_scenarios_snapshots()
            self.process_scene_capture_request()

            return

    def run_scenario_for_scene_capture(self, wid: int, scenario_id: int):
        worker : BACREWorkerRef = self.workers[wid]
        
        worker.scenario = copy.deepcopy(self.get_generated_scenario_from_id(scenario_id))
        worker.scenario.outcomes.clear()
        worker.run_scenario_request = self.scenario_builder.create_default_run_scenario_request()
        worker.run_scenario_request.scene_description = worker.scenario.scene_description

        worker.run_scenario_request.take_scene_capture = True
        worker.run_scenario_request.scene_capture_only = True
        worker.run_scenario_request.scene_capture_settings = copy.deepcopy(self.scene_capture_settings)

        self.log("info", f"Recreating scenario for scene captures for Worker {wid} / Scenario ID {scenario_id}")
        self.submit_run_scenario_request(wid)

    def run_scene_capture_step(self, wid: int):
        if not self.mode[self.MODE_SCENE_CAPTURE]:
            return
        
        # Only workers with rosbridge nodes running on same computer as the client can perform scene captures due to the large image sizes
        if wid not in self.local_wids:
            return
        
        if not self.are_all_vehicle_nodes_ready(wid):
            return
        
        worker : BACREWorkerRef = self.workers[wid]
            
        if wid not in self.worker_mspids.keys():
            if self.num_mspids_remaining() > 0 and not worker.b_waiting_for_analyze_scenario_request:
                self.log("info", "-" * 30 + f"SCENE CAPTURE STEP (wid {wid})" + "-" * 30, b_log_ros=False)
                self.worker_mspids[wid] = self.get_next_mspid()
                scenario_id = self.worker_mspids[wid]
                self.run_scenario_for_scene_capture(wid, scenario_id)
                return
            else:
                return
                
        # Block until ready to analyze scenario
        if worker.b_waiting_for_analyze_scenario_request or worker.analyze_scenario_request is None:
            return

        self.log("info", "-" * 30 + f'SCENE CAPTURE STEP (wid {wid})' + "-" * 30, b_log_ros=False)
        if not self.process_analyze_scenario_request(wid):
            return
        
        self.mspids_completed.append(self.worker_mspids[wid])
        self.log("info", f"Num. Captures Completed: {len(self.mspids_completed)} / {len(self.mspids)}")
        
        worker.reset()
        self.worker_mspids.pop(wid)

        # Check if done creating scene captures
        if self.num_mspids_remaining() > 0:
            self.worker_mspids[wid] = self.get_next_mspid()
            scenario_id = self.worker_mspids[wid]
            self.run_scenario_for_scene_capture(wid, scenario_id)
            return
        else:
            self.log("info", f"Worker {wid} done with scene capture")
        
        if len(self.mspids_completed) == len(self.mspids):
            self.mode[self.MODE_SCENE_CAPTURE] = False
            self.log("info", f"Done with scene capture mode")

            if self.b_in_testing_mode:
                self.mode[self.MODE_REPLAY] = True
                self.plot_max_regret_scene()
                self.plot_max_regret_scenes()
                self.plot_max_regret_scenario_tree()
                self.process_replay_request()
            return

    def main_step(self, wid: int):
        """Overriden function from AutoSceneGenClient. Will be called automatically from AutoSceneGenClient.main_loop_timer_cb().
        Call the appropriate function based on the current mode of operation.
        """
        self.b_done_testing = not self.mode[self.MODE_BACRE] and not self.mode[self.MODE_REPLAY] and not self.mode[self.MODE_RANDOM] and not self.mode[self.MODE_CGA] and not self.mode[self.MODE_SCENE_CAPTURE]
        if self.b_done_testing:
            self.log("info", "Done testing")
            return

        worker : BACREWorkerRef = self.workers[wid]

        # Worker must be registered
        if not worker.b_registered_with_asg_client:
            return

        # Check if the worker stopped receiving RunScenario requests
        if wid in self.worker_mspids.keys() and worker.num_resubmission_attempts >= 2:
            self.log("info", f"Worker {wid} stopped receiving RunScenario requests. Ignoring worker.")
            self.worker_mspids.pop(wid)

            # Only reset the worker if this one case holds, otherwise BACRE can't finish creating the initial population
            if self.mode[self.MODE_BACRE] and worker.b_running_base_scenario:
                self.log("info", f"Resetting Worker {wid} so BACRE can create initial population.")
                worker.reset()

        # Ignore worker if it stopped receiving RunScenario requests
        if worker.num_resubmission_attempts >= 2:
            return
        
        # Scene capture mode
        if self.mode[self.MODE_SCENE_CAPTURE]:
            self.run_scene_capture_step(wid)
            return
        
        # Comes AFTER scene capture mode, since we don't need any vehicle nodes for scene captures
        # Check if any vehicle nodes unregistered while running a scenario
        if wid in self.worker_mspids.keys() and len(worker.registered_vehicle_nodes) != self.num_vehicle_nodes:
            self.log("info", f"A vehicle node for Worker {wid} unregistered while running a scenario. Resetting worker.")
            worker.reset()
            self.worker_mspids.pop(wid)
        
        # Continuous GA mode
        if self.mode[self.MODE_CGA]:
            self.run_cga_step(wid)
            return
        
        if len(worker.registered_vehicle_nodes) != self.num_vehicle_nodes:
            return

        # BACRE mode
        if self.mode[self.MODE_BACRE]:
            self.run_bacre_step(wid)

        # Replay mode
        elif self.mode[self.MODE_REPLAY]:
            self.run_replay_step(wid)

        # Random mode
        elif self.mode[self.MODE_RANDOM]:
            self.run_random_step(wid)


def main(args=None):
    # Place the main directory inside the src folder
    path_prefix = os.getcwd()
    if "/src" not in path_prefix:
        path_prefix = os.path.join(path_prefix, "src")
    path_prefix = os.path.join(path_prefix, "bacre_data", 'astar_trav_2D', f"iterative_experiments")
    main_dir = os.path.join(path_prefix, "sunyaw0_rcr0.05_av4_run1") # Folder for storing all data w.r.t. an individual experiment

    if not os.path.isdir(main_dir):
        os.makedirs(main_dir) 

    ssa_config = [
        StructuralSceneActorConfig(blueprint_directory="/Game/Blueprints/Structural_Scene_Actors/Trees", blueprint_name="BP_SSA_Tree_Juniper_Rocky_Mountain_Field_Desktop", num_instances=6, max_scale=1., ssa_type="tree"),
        # StructuralSceneActorConfig(blueprint_directory="/Game/Blueprints/Structural_Scene_Actors/Trees", blueprint_name="BP_SSA_Tree_Juniper_Rocky_Mountain_Field_Hero", num_instances=6, max_scale=1., ssa_type="tree"),
        StructuralSceneActorConfig(blueprint_directory="/Game/Blueprints/Structural_Scene_Actors/Bushes", blueprint_name="BP_SSA_Bush_Barberry_Desktop2", num_instances=6, max_scale=1., ssa_type="bush"),
        # StructuralSceneActorConfig(blueprint_directory="/Game/Blueprints/Structural_Scene_Actors/Bushes", blueprint_name="BP_SSA_Bush_Barberry_Desktop4", num_instances=6, max_scale=1., ssa_type="bush"),
    ]

    landscape_nominal_size = 60.
    landscape_size = np.array([landscape_nominal_size, landscape_nominal_size])

    ssa_attr = StructuralSceneActorAttributes(
        x = {"default": 0., "range": (0., landscape_size[0])},  # [m]
        y = {"default": 0., "range": (0., landscape_size[1])},  # [m]
        yaw = 0.,   # [deg]
        scale = 1.
    )

    txt_attr = TexturalAttributes(
        sunlight_inclination = 30.,   # [deg]
        sunlight_yaw = 0.             # [deg]
    )

    opr_attr = OperationalAttributes(
        start_location_x = 10.,   # [m]
        start_location_y = 10.,   # [m]
        start_yaw = 45.,          # [deg]
        goal_location_x = 50.,    # [m]
        goal_location_y = 50.,    # [m]
        goal_radius = 5.,         # [m]
        sim_timeout_period = 17.,               # [s]
        vehicle_idling_timeout_period = 5.,     # [s]
        vehicle_stuck_timeout_period = 5.,      # [s]
        max_vehicle_roll = 70.,                 # [deg]
        max_vehicle_pitch = 70.,                # [deg]
    )

    astar_path_planner_dict = {
        'env_size': landscape_size,
        'border_size': 0.,
        'minor_cell_size': 0.25,
        'major_cell_size': 0.5,
        'bell_curve_sigma': 2.5, # Usually 2.5
        'bell_curve_alpha': 1/50.,
        'max_votes_per_vertex': 50,
        'radius_threashold1': 3.,
        'radius_threashold2': 6.,
        'use_three_point_edge_cost': True,
    }

    scenario_builder = ScenarioBuilderAndRefAgent(
        landscape_nominal_size = landscape_nominal_size,         # The side-length of the landscape in [m], this is a square landscape
        landscape_subdivisions = 1,         # The number of times the two base triangles in the nominal landscape should be subdivided
        landscape_border = 100.,         # Denotes the approximate length to extend the nominal landscape in [m]
        ssa_attr = ssa_attr,
        ssa_config = ssa_config,                # Information about the SSAs.
        b_ssa_casts_shadow = True,            # Indicates if the SSAs cast a shadow in the game
        b_allow_collisions = True,            # Indicate if simulation keeps running in the case of vehicle collisions
        txt_attr = txt_attr,
        opr_attr = opr_attr,
        start_obstacle_free_radius = 10.,     # Obstacle-free radius [m] around the start location
        goal_obstacle_free_radius = 10.,      # Obstacle-free radius [m] around the goal location
        astar_path_planner_dict = astar_path_planner_dict,
        nominal_vehicle_speed = 5.,
        opt_agent_min_obstacle_proximity = 4.,
    )

    # Default scene capture settings
    scene_capture_settings = auto_scene_gen_msgs.SceneCaptureSettings()
    scene_capture_settings.image_size = 1024

    scene_capture_settings.draw_annotations = False
    scene_capture_settings.goal_sphere_thickness = 5.
    scene_capture_settings.goal_sphere_color = std_msgs.ColorRGBA(r=0., g=0., b=1., a=1.) # Blue

    scene_capture_settings.ortho_aerial = True
    scene_capture_settings.perspective_aerial = False
    scene_capture_settings.aerial_padding = [0, 10, 20]

    scene_capture_settings.front_aerial = True
    scene_capture_settings.left_front_aerial = False
    scene_capture_settings.left_aerial = False
    scene_capture_settings.left_rear_aerial = False
    scene_capture_settings.rear_aerial = False
    scene_capture_settings.right_rear_aerial = False
    scene_capture_settings.right_aerial = False
    scene_capture_settings.right_front_aerial = False
    scene_capture_settings.vehicle_start_pov = True
    scene_capture_settings.vehicle_start_rear_aerial = True

    auto_scene_gen_client_dict = {
        "node_name": "bacre_2D_astar_trav",
        "main_dir": main_dir,
        "asg_client_name": "asg_client",        # Name of the AutoSceneGenClient
        "num_vehicle_nodes": 3,                 # Number of vehicle nodes in the vehicle's the autonomy stack
        "num_workers": 12,                      # Number of AutoSceneGenWorkers
        "base_wid": 0,                          # Base (lowest) AutoSceneGenWorker ID
        "local_wids": list(range(6)),
        "worker_class": BACREWorkerRef,            # Class of type AutoSceneGenWorker, used to instantiate the worker references
        "scenario_builder": scenario_builder,
    }
    
    bacre_dict = {
        "mode": BACRE2DAstarTrav.MODE_BACRE,         # Modes of operation
        "num_runs_per_scenario": 3,          # Number of times we run each scenario for to predict the difficulty for the test vehicle
        "num_base_env": 200,                               # Number of base or "seed" environments to create
        "num_env_to_generate": 5000,                       # Maximum number of environments to generate
        "num_mutations": 1,
        "b_regret_uses_equivalent_x_errors": True,
        "initial_prob_editing_random_env": 0.1, # .5,
        "prob_editing_random_env": 0.1,                 # Probability of choosing a random parent environment to edit (over all generated levels)
        "upper_search_quantile": 0.01,                   # Specifies the upper quantile of high-regret scenarios from which to randomly choose from
        "initial_regret_curation_ratio": 0.05, # .5             # Specifies the initial regret curation ratio
        "regret_curation_ratio": 0.05,             # Specifies the desired regret curation ratio
        "anneal_duration": 1000,
        "parent_selection_method": BACRE2DAstarTrav.PAR_SELECT_RCR,  # Method for selecting a scenario's parent
        "ssa_add_prob": 0.3,
        "ssa_remove_prob": 0.3,
        "ssa_perturb_prob": 0.4,
        "data_save_freq": 100,                            # Save data manager at this frequency of generated scenarios
        "num_difficult_scenarios_to_plot": 50,           # Number of most difficult scenarios to plot
        "cga_population_size": 200,
        "cga_num_families_per_population": 10,
        "cga_num_populations": 25,
        "cga_mutation_prob": 0.2,
        "scene_capture_settings": scene_capture_settings,
        "scene_capture_request": BACRE2DAstarTrav.SCR_BEST_FAMILY_TREE,
        "scene_capture_request_value": 50,
        "replay_request_rank_range": [1,10],
        "replay_request_num_replays": 3,
    }

    b_plot_group_summary = False
    group_summary_prefix = "bacre_rand"
    group_summary_dict = { # Key = Experiment Name, Value = (list of experiment folders, color)
        "Random": (("sunyaw0_rand_av4_run1", "sunyaw0_rand_av4_run2", "sunyaw0_rand_av4_run3", "sunyaw0_rand_av4_run4", "sunyaw0_rand_av4_run5"), "red"),
        "BACRE": (("sunyaw0_rcr0.05_av4_run1", "sunyaw0_rcr0.05_av4_run2", "sunyaw0_rcr0.05_av4_run3", "sunyaw0_rcr0.05_av4_run4", "sunyaw0_rcr0.05_av4_run5"), "blue"),
        "BACRE w/\nAnnealing": (("sunyaw0_rcr0.05_anneal_av4_run1", "sunyaw0_rcr0.05_anneal_av4_run2", "sunyaw0_rcr0.05_anneal_av4_run3", "sunyaw0_rcr0.05_anneal_av4_run4", "sunyaw0_rcr0.05_anneal_av4_run5"), "green"),
        "BACRE w/\n3 Mutations": (("sunyaw0_rcr0.05_edits3_av4_run1", "sunyaw0_rcr0.05_edits3_av4_run2", "sunyaw0_rcr0.05_edits3_av4_run3", "sunyaw0_rcr0.05_edits3_av4_run4", "sunyaw0_rcr0.05_edits3_av4_run5"), "black"),
        "BACRE w/\nAnnealing + NEET": (("sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run1","sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run2", "sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run3", "sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run4", "sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run5"), "orange"),
    }

    b_plot_cga_group_summary = False
    cga_group_summary_dict = { # Key = Experiment Name, Value = (list of experiment folders, color)
        "CGA (6 Obstacles)": (("sunyaw0_rcr0.05_cga_tree3_bush3_av4_run1", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run2", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run3", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run4", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run5"), "blue"),
        "CGA (12 Obstacles)": (("sunyaw0_rcr0.05_cga_tree6_bush6_av4_run1", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run2", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run3", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run4", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run5"), "red"),
    }

    b_make_group_summary_log = False
    group_summary_log_prefix = "bacre_rand_cga"
    regret_thresholds = (0.375, 0.5, 0.75, 1.)
    group_summary_log_dict = {
        "Random": ("sunyaw0_rand_av4_run1", "sunyaw0_rand_av4_run2", "sunyaw0_rand_av4_run3", "sunyaw0_rand_av4_run4", "sunyaw0_rand_av4_run5"),
        "BACRE": ("sunyaw0_rcr0.05_av4_run1", "sunyaw0_rcr0.05_av4_run2", "sunyaw0_rcr0.05_av4_run3", "sunyaw0_rcr0.05_av4_run4", "sunyaw0_rcr0.05_av4_run5"),
        "BACRE w/Annealing": ("sunyaw0_rcr0.05_anneal_av4_run1", "sunyaw0_rcr0.05_anneal_av4_run2", "sunyaw0_rcr0.05_anneal_av4_run3", "sunyaw0_rcr0.05_anneal_av4_run4", "sunyaw0_rcr0.05_anneal_av4_run5"),
        "BACRE w/3 Mutations": ("sunyaw0_rcr0.05_edits3_av4_run1", "sunyaw0_rcr0.05_edits3_av4_run2", "sunyaw0_rcr0.05_edits3_av4_run3", "sunyaw0_rcr0.05_edits3_av4_run4", "sunyaw0_rcr0.05_edits3_av4_run5"),
        "BACRE w/Annealing + NEET": ("sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run1","sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run2", "sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run3", "sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run4", "sunyaw0_rcr0.05_no-equiv-error_anneal_av4_run5"),
        "CGA (6 Obstacles)": ("sunyaw0_rcr0.05_cga_tree3_bush3_av4_run1", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run2", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run3", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run4", "sunyaw0_rcr0.05_cga_tree3_bush3_av4_run5"),
        "CGA (12 Obstacles)": ("sunyaw0_rcr0.05_cga_tree6_bush6_av4_run1", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run2", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run3", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run4", "sunyaw0_rcr0.05_cga_tree6_bush6_av4_run5"),
    }

    def all_elements_are_numbers(data):
        for x in data:
            if not (isinstance(x, int) or isinstance(x,float)):
                return False
        return True
    
    def write_dict_to_txt(file: io.TextIOWrapper, param_dict):
        for key,value in param_dict.items():
            if isinstance(value, dict):
                file.write(f"- {key}:\n")
                for key2,value2 in value.items():
                    f.write(f" "*5 + f"- {key2}: {value2}\n")
            elif (isinstance(value, list) or isinstance(value, tuple)) and not all_elements_are_numbers(value):
                file.write(f"- {key}:\n")
                for value2 in value:
                    file.write(f" "*5 + f"- {value2}\n")
            else:
                file.write(f"- {key}: {value}\n")

    # Write parameters to txt file
    if not os.path.exists(os.path.join(main_dir, "parameters.txt")):
        parameters_file = os.path.join(main_dir, "parameters.txt") # Original parameters get written here, in case we need to refer back to them
    else:
        parameters_file = os.path.join(main_dir, "parameters_latest.txt") # Write current parameters here
    with open(parameters_file, "w") as f:
        f.write("#################### AutoSceneGenClient Paremters: ####################\n")
        write_dict_to_txt(f, auto_scene_gen_client_dict)
        
        f.write("\n#################### BACRE Paremters: ####################\n")
        write_dict_to_txt(f, bacre_dict)
        f.write("\n")

    scenario_builder.log_parameters_to_file(parameters_file)

    rclpy.init(args=args)
    node = BACRE2DAstarTrav(auto_scene_gen_client_dict, **bacre_dict)

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2) # Use multithreaded executor so we can allow for service calls to execute while main loop is calculting optimal trajectories
    executor.add_node(node)

    # By placing the spin() call in a try-except-finally block if the user presses ctrl-c, we can buy enough time to have the AutoSceneGenClient
    # publish its updated status to all workers.
    try:
        # node.make_main_figure_plots()
        b_spin = True
        if b_plot_group_summary:
            node.plot_group_summary(group_summary_dict, group_summary_prefix, num_legend_cols=3, legend_fontsize=12)
            b_spin = False
        if b_plot_cga_group_summary:
            node.plot_cga_group_summary(cga_group_summary_dict)
            b_spin = False
        if b_make_group_summary_log:
            node.group_summary_logging(group_summary_log_dict, regret_thresholds, group_summary_log_prefix)
            b_spin = False
        if b_spin:
            while executor.context.ok():
                executor.spin_once()
                if node.b_done_testing:
                    break
    except KeyboardInterrupt:
        node.log("info", "Keyboard interrupt. Shutting down...")
    except Exception as e:
        node.log("error", f"EXCEPTION: {traceback.format_exc()}") # Print entire exception and source of problem
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()