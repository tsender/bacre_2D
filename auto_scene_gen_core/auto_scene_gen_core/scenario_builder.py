import copy
import math
import numpy as np
import os
from typing import Tuple, List, Dict, Union

import auto_scene_gen_msgs.msg as auto_scene_gen_msgs
import auto_scene_gen_msgs.srv as auto_scene_gen_srvs

def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))


class ScenarioAttribute:
    """The primary class that contains information about a specific scenario attribute"""

    def __init__(self, name: str, value):
        """
        Args:
            - name: Attribute name
            - value: Describes the value of the attribute. Can be a scalar or a dict. If a dict, then you must provide the keys "default" and "range".
        """
        self.name = name
        self.default = None # Default value
        self.range = None   # Allowed range of values (must include the default value)
        self._process_value(name, value)

    def __str__(self):
        return str({"name": self.name, "default": self.default, "range": self.range})

    def _process_value(self, name: str, value):
        """Process the attribute value. Raise a ValueError if there is a problem.

        Args:
            - name: Attribute name
            - value: The value of the attribute, can be a scalar or a dict.
        """
        # Input is scalar means this is the default (and only) value allowed
        if isinstance(value, int) or isinstance(value, float):
            self.default = value
            return

        if not isinstance(value, dict):
            raise ValueError(f"Value for attribute '{name}' must be a dict, int, or float. Received type {type(value)}.")

        # Input is a dict and must define the default value and allowed range of values
        if "default" not in value.keys() or "range" not in value.keys():
            raise ValueError(f"The keys 'default' and 'range' must be provided in the dict for attribute '{name}'.")

        default = value["default"]
        vrange = value["range"] # Avoid name-clash with class range

        # Default value must be int or float
        if not isinstance(default, int) and not isinstance(default, float):
            raise ValueError(f"The 'default' field in '{name}' dict must be an int or float.")

        # If range is a tuple/list, make sure values are valid
        if isinstance(vrange, tuple) or isinstance(vrange, list):
            if not len(vrange) == 2:
                raise ValueError(f"The 'range' field in '{name}' dict must have two elements in it.")

            if vrange[0] > vrange[1]:
                raise ValueError(f"Invalid 'range' field for '{name}' dict. {vrange[0]} is not <= {vrange[1]}. Values must be in increasing order.")
            
            if default < vrange[0] or vrange[1] < default:
                raise ValueError(f"Invalid 'range' field for '{name}' dict. 'default' value must lie within bounds set by 'range' field.")
            
            self.default = default
            self.range = (vrange[0], vrange[1])
            return
        else:
            raise ValueError(f"Unrecognized value for 'range' field for '{name}' dict. Must be a tuple/list of the form (a,b) where a <= b.")
    
    def random(self):
        """Get random value from the attribute's range. Return default value if no range exists."""
        if self.range is None:
            return self.default
        else:
            return np.random.uniform(self.range[0], self.range[1])

class ScenarioAttributeGroup:
    """The base class from which we can define the various attribute groupings in our scenario. All ScenarioAttributes objects are stored both as a
    class variable and also inside a dictionary to provide ease of access depending on the user's needs."""

    def __init__(self):
        self.attrs : Dict[str, ScenarioAttribute] = {}  # Dictionary of all ScenarioAttribute instances in the class
        self.fixed_attrs = []                           # Names of the fixed attributes
        self.var_attrs = []                             # Names of the variable attributes
    
    def _process_attr(self, name: str, value):
        """Process the attribute value

        Args:
            - name: Attribute name
            - value: The value of the attribute

        Returns:
            - The ScenarioAttribute
        """
        attr = ScenarioAttribute(name, value)
        if attr.range is None:
            self.fixed_attrs.append(name)
        else:
            self.var_attrs.append(name)
        self.attrs[name] = attr
        return attr

    def get_default(self, name):
        """Get default value for an attribute by name"""
        return self.attrs[name].default
    
    def get_range(self, name):
        """Get range of values for an attribute by name"""
        return self.attrs[name].range
    
    def get_random(self, name):
        """Get a random valid value for an attribute by name"""
        return self.attrs[name].random()
    

class StructuralSceneActorAttributes(ScenarioAttributeGroup):
    # Attribute names
    X = "x"
    Y = "y"
    YAW = "yaw"
    SCALE = "scale"

    def __init__(self,
                 x: Union[float, dict],     # The x-coordinate for the SSA [m]
                 y: Union[float, dict],     # The y-coordinate for the SSA [m]
                 yaw: Union[float, dict],   # The yaw angle for the SSA [deg]
                 scale: Union[float, dict]  # The scale factor for the SSA (applies to all three dimensions) multilying the max_scale parameter for the particular SSA (sse StructuralSceneActorConfig)
                 ):
        super().__init__()
        self.x = self._process_attr(self.X, x)
        self.y = self._process_attr(self.Y, y)
        self.yaw = self._process_attr(self.YAW, yaw)
        self.scale = self._process_attr(self.SCALE, scale)


class TexturalAttributes(ScenarioAttributeGroup):
    # Attribute names
    SUNLIGHT_INCLINATION = "sunlight_inclination"
    SUNLIGHT_YAW = "sunlight_yaw"

    def __init__(self,
                 sunlight_inclination: Union[float, dict],  # The angle the sun makes with the horizontal plane [deg]
                 sunlight_yaw: Union[float, dict]           # The angle in which the sun is pointing in [deg]
                 ):
        super().__init__()
        self.sunlight_inclination = self._process_attr(self.SUNLIGHT_INCLINATION, sunlight_inclination)
        self.sunlight_yaw = self._process_attr(self.SUNLIGHT_YAW, sunlight_yaw)


class OperationalAttributes(ScenarioAttributeGroup):
    # Attribute names
    START_LOCATION_X = "start_location_x"
    START_LOCATION_Y = "start_location_y"
    START_YAW = "start_yaw"
    GOAL_LOCATION_X = "goal_location_x"
    GOAL_LOCATION_Y = "goal_location_y"
    GOAL_RADIUS = "goal_radius"

    SIM_TIMEOUT_PERIOD = "sim_timeout_period"
    VEHICLE_IDLING_TIMEOUT_PERIOD = "vehicle_idling_timeout_period"
    VEHICLE_STUCK_TIMEOUT_PERIOD = "vehicle_stuck_timeout_period"

    MAX_VEHICLE_ROLL = "max_vehicle_roll"
    MAX_VEHICLE_PITCH = "max_vehicle_pitch"

    def __init__(self,
                 start_location_x: Union[float, dict],              # The x-coordinate for the vehicle's starting position [m]
                 start_location_y: Union[float, dict],              # The y-coordinate for the vehicle's starting position [m]
                 start_yaw: Union[float, dict],                     # The yaw angle for the vehicle's starting position [deg]
                 goal_location_x: Union[float, dict],               # The x-coordinate for the vehicle's goal position [m]
                 goal_location_y: Union[float, dict],               # The y-coordinate for the vehicle's goal position [m]
                 goal_radius: Union[float, dict],                   # The goal radius [m]
                 sim_timeout_period: Union[float, dict],            # The simulation timeout period [s]
                 vehicle_idling_timeout_period: Union[float, dict], # The maximum amount of time the vehicle can idle [sec]
                 vehicle_stuck_timeout_period: Union[float, dict],  # The maximum amount of time the vehicle can be stuck [s]
                 max_vehicle_roll: Union[float, dict],              # The maximum allowed roll angle for the vehicle [deg]
                 max_vehicle_pitch: Union[float, dict],             # The maximum allowed pitch angle for the vehicle [deg]
                 ):
        super().__init__()
        self.start_location_x = self._process_attr(self.START_LOCATION_X, start_location_x)
        self.start_location_y = self._process_attr(self.START_LOCATION_Y, start_location_y)
        self.start_yaw = self._process_attr(self.START_YAW, start_yaw)
        self.goal_location_x = self._process_attr(self.GOAL_LOCATION_X, goal_location_x)
        self.goal_location_y = self._process_attr(self.GOAL_LOCATION_Y, goal_location_y)
        self.goal_radius = self._process_attr(self.GOAL_RADIUS, goal_radius)

        self.sim_timeout_period = self._process_attr(self.SIM_TIMEOUT_PERIOD, sim_timeout_period)
        self.vehicle_idling_timeout_period = self._process_attr(self.VEHICLE_IDLING_TIMEOUT_PERIOD, vehicle_idling_timeout_period)
        self.vehicle_stuck_timeout_period = self._process_attr(self.VEHICLE_STUCK_TIMEOUT_PERIOD, vehicle_stuck_timeout_period)

        self.max_vehicle_roll = self._process_attr(self.MAX_VEHICLE_ROLL, max_vehicle_roll)
        self.max_vehicle_pitch = self._process_attr(self.MAX_VEHICLE_PITCH, max_vehicle_pitch)

    def get_default_start_location(self):
        """Return the default start location as a tuple of (x,y)"""
        return np.array([self.start_location_x.default, self.start_location_y.default])
    
    def get_default_goal_location(self):
        """Return the default goal location as a tuple of (x,y)"""
        return np.array([self.goal_location_x.default, self.goal_location_y.default])


class StructuralSceneActorConfig:
    """Describes additional information about each SSA that will be placed in the scene"""

    def __init__(self, 
                 blueprint_directory: str,  # The directory to find the Blueprint in the UE project, starts with "/Game/"
                 blueprint_name: str,       # The name of the Blueprint (exluding extensions)
                 num_instances: int,        # The number of instances that can be placed in the game
                 max_scale: float,          # The maximum scale factor (we keep this separate from the scene attributes in case the user wants more control)
                 ssa_type: str              # The type of SSA, e.g., tree or bush
                 ):
        self.path_name = os.path.join(blueprint_directory, f"{blueprint_name}.{blueprint_name}_C") # The full path name to the BP in the UE project
        self.num_instances = num_instances
        self.max_scale = max_scale
        self.ssa_type = ssa_type
    
    def __str__(self):
        d = {"path_name": self.path_name, "num_instances": self.num_instances, "max_scale": self.max_scale, "ssa_type": self.ssa_type}
        return str(d)
    

class SSAChangeLogAttributes:
    """SSA attributes that can be modified"""
    def __init__(self):
        self.x = None
        self.y = None
        self.yaw = None
        self.scale = None
        self.b_visible = None


class SSAChangeLog:
    """Class for defining the changes for a specific SSA"""
    def __init__(self):
        self.ssa_idx = None # SSA index in the StructuralSceneActorConfig object
        self.obs_idx = None # Object index within the SSA layout of the RunScenario request
        self.old : SSAChangeLogAttributes = SSAChangeLogAttributes()
        self.new : SSAChangeLogAttributes = SSAChangeLogAttributes()
    

class AutoSceneGenScenarioBuilder:
    """Base class for creating scenarios to run with the UE4 AutomaticSceneGeneration plugin
    
    NOTE: This class expects all position measurements to be in [m] and expressed in a right-handed north-west-up coordinate frame.
    This class will handle the conversion to UE4 internally.
    """
    def __init__(self,
                landscape_nominal_size: float,                   # The side-length of the landscape in [m], this is a square landscape
                landscape_subdivisions: int,                    # The number of times the two base triangles in the nominal landscape should be subdivided
                landscape_border: float,                        # Denotes the approximate length to extend the nominal landscape in [m]
                ssa_attr: StructuralSceneActorAttributes,       # The StructuralSceneActorAttributes instance to use
                ssa_config: List[StructuralSceneActorConfig],   # A list of `StructuralSceneActorConfig` objects for the various SSAs that can be placed in the scene
                b_ssa_casts_shadow: bool,                       # Indicates if the SSAs cast a shadow in the game
                b_allow_collisions: bool,                       # Indicate if simulation keeps running in the case of vehicle collisions
                txt_attr: TexturalAttributes,                   # The TexturalAttributes instance to use
                opr_attr: OperationalAttributes,                # The OperationalAttributes instance to use
                start_obstacle_free_radius: float,              # Obstacle-free radius [m] around the start location
                goal_obstacle_free_radius: float,              # Obstacle-free radius [m] around the goal location
                ):
        # Landscape info
        self.landscape_nominal_size = landscape_nominal_size
        self.landscape_size = np.array([landscape_nominal_size, landscape_nominal_size])
        self.landscape_subdivisions = landscape_subdivisions
        self.landscape_border = landscape_border

        # Structural scene actor (SSA) params
        self.ssa_attr = ssa_attr
        self.ssa_config = ssa_config
        self.b_allow_collisions = b_allow_collisions
        self.b_ssa_casts_shadow = b_ssa_casts_shadow
        self.num_ssa_subclasses = len(ssa_config)
        self.num_ssa_instances = 0
        for elem in ssa_config:
            self.num_ssa_instances += elem.num_instances

        # Textural attributes
        self.txt_attr = txt_attr

        # Operational attributes
        self.opr_attr = opr_attr
        self.start_obstacle_free_radius = start_obstacle_free_radius
        self.goal_obstacle_free_radius = goal_obstacle_free_radius

    def log_parameters_to_file(self, file_path: str):
        """Append all parameters to a provided txt file. Used for logging purposes.
        
        Args:
            - file: The .txt file to append data to
        """
        with open(file_path, "a") as f:
            f.write("#################### Scenario Builder Paremters: ####################\n")
            f.write(f"- landscape_nominal_size: {self.landscape_nominal_size}\n")
            f.write(f"- landscape_subdivisions: {self.landscape_subdivisions}\n")
            f.write(f"- landscape_border: {self.landscape_border}\n")

            f.write(f"- ssa_attr:\n")
            for key,value in self.ssa_attr.attrs.items():
                f.write(f" "*5 + f"- {str(value)}\n")
            
            f.write(f"- ssa_config:\n")
            for value in self.ssa_config:
                f.write(f" "*5 + f"- {value}\n")

            f.write(f"- b_ssa_casts_shadow: {self.b_ssa_casts_shadow}\n")
            f.write(f"- b_allow_collisions: {self.b_allow_collisions}\n")

            f.write(f"- txt_attr:\n")
            for key,value in self.txt_attr.attrs.items():
                f.write(f" "*5 + f"- {str(value)}\n")

            f.write(f"- opr_attr:\n")
            for key,value in self.opr_attr.attrs.items():
                f.write(f" "*5 + f"- {str(value)}\n")

            f.write(f"- start_obstacle_free_radius: {self.start_obstacle_free_radius}\n")
            f.write(f"- goal_obstacle_free_radius: {self.goal_obstacle_free_radius}\n")

    def create_default_run_scenario_request(self):
        """Create default RunScenario request"""
        request = auto_scene_gen_srvs.RunScenario.Request()
        request.vehicle_start_location.x = self.opr_attr.start_location_x.default
        request.vehicle_start_location.y = self.opr_attr.start_location_y.default
        request.vehicle_start_yaw = self.opr_attr.start_yaw.default

        request.vehicle_goal_location.x = self.opr_attr.goal_location_x.default
        request.vehicle_goal_location.y = self.opr_attr.goal_location_y.default
        request.goal_radius = self.opr_attr.goal_radius.default

        request.sim_timeout_period = self.opr_attr.sim_timeout_period.default
        request.vehicle_idling_timeout_period = self.opr_attr.vehicle_idling_timeout_period.default
        request.vehicle_stuck_timeout_period = self.opr_attr.vehicle_stuck_timeout_period.default
        request.max_vehicle_roll = self.opr_attr.max_vehicle_roll.default
        request.max_vehicle_pitch = self.opr_attr.max_vehicle_pitch.default
        request.allow_collisions = self.b_allow_collisions

        request.take_scene_capture = False
        request.scene_capture_only = False

        request.scene_description = self.create_default_scene_description_msg()
        return request
    
    def create_default_scene_description_msg(self):
        """Create default SceneDescription message"""
        scene_description = auto_scene_gen_msgs.SceneDescription()

        landscape = auto_scene_gen_msgs.LandscapeDescription()
        landscape.nominal_size = self.landscape_nominal_size
        landscape.subdivisions = self.landscape_subdivisions
        landscape.border = self.landscape_border
        scene_description.landscape = landscape
        
        scene_description.sunlight_inclination = self.txt_attr.sunlight_inclination.default
        scene_description.sunlight_yaw_angle = self.txt_attr.sunlight_yaw.default

        ssa_array = []
        for i in range(len(self.ssa_config)):
            n = self.ssa_config[i].num_instances
            max_scale = self.ssa_config[i].max_scale

            layout = auto_scene_gen_msgs.StructuralSceneActorLayout()
            layout.path_name = self.ssa_config[i].path_name
            layout.num_instances = n
            layout.visible = [False] * n
            layout.cast_shadow = [self.b_ssa_casts_shadow] * n
            x = []
            y = []
            yaw = []
            scale = []
            
            x += [self.ssa_attr.x.default] * n
            y += [self.ssa_attr.y.default] * n
            yaw += [self.ssa_attr.yaw.default] * n
            scale += [self.ssa_attr.scale.default * max_scale] * n

            # layout.* are arrays, not lists.
            layout.x = x
            layout.y = y
            layout.yaw = yaw
            layout.scale = scale
            ssa_array.append(layout)

        scene_description.ssa_array = ssa_array
        return scene_description
    
    def get_unreal_engine_run_scenario_request(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        """Return the equivalent UE RunScenario request
        
        Args:
            - scenario_request: Input RunScenario request

        Returns:
            - The UE RunScenario request
        """
        ue_request = copy.deepcopy(scenario_request)

        ue_request.vehicle_start_location.x = scenario_request.vehicle_start_location.x * 100. # Put in [cm]
        ue_request.vehicle_start_location.y = scenario_request.vehicle_start_location.y * -100. # Put in [cm]
        ue_request.vehicle_start_yaw = -scenario_request.vehicle_start_yaw

        ue_request.vehicle_goal_location.x = scenario_request.vehicle_goal_location.x * 100. # Put in [cm]
        ue_request.vehicle_goal_location.y = scenario_request.vehicle_goal_location.y * -100. # Put in [cm]
        ue_request.goal_radius = scenario_request.goal_radius * 100. # Put in [cm]

        ue_request.scene_description = self.get_unreal_engine_scene_description(scenario_request.scene_description)

        return ue_request
    
    def get_unreal_engine_scene_description(self, scene_description: auto_scene_gen_msgs.SceneDescription):
        """Return the equivalent UE scene description
        
        Args:
            - scene_description: The input scene description

        Returns:
            - The UE scene description
        """
        ue_scene_description = copy.deepcopy(scene_description)
        ue_scene_description.landscape.nominal_size = scene_description.landscape.nominal_size * 100.
        ue_scene_description.landscape.border = scene_description.landscape.border * 100.

        ue_scene_description.sunlight_yaw_angle = -scene_description.sunlight_yaw_angle

        for layout in ue_scene_description.ssa_array:
            layout.x = (np.array(layout.x) * 100.).tolist()
            layout.y = (np.array(layout.y) * -100.).tolist()
            layout.yaw = (np.array(layout.yaw) * -1.).tolist()

        return ue_scene_description
    
    def is_scenario_request_feasible(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        """Indicates if the scenario request is feasible. Default is True. Can be overridden in a child class if you have feasibility requirements."""
        return True
    
    def get_num_active_ssa(self, ssa_array: List[auto_scene_gen_msgs.StructuralSceneActorLayout]):
        """Return the number of active SSAs in the given scene description
        
        Args:
            - ssa_array: Input SSA array with n types

        Returns:
            - Total number of active SSAs across the n types
            - Tuple, beakdown of the number of active SSAs for each of the n types"""
        if ssa_array is None or len(ssa_array) == 0:
            return None
        else:
            total = 0
            num_active_per_ssa = []
            for layout in ssa_array:
                n = int(np.sum(layout.visible))
                num_active_per_ssa.append(n)
                total += n
            return total, tuple(num_active_per_ssa)

    def get_num_inactive_ssa(self, ssa_array: List[auto_scene_gen_msgs.StructuralSceneActorLayout]):
        """Return the number of inactive SSAs
        
        Args:
            - ssa_array: Input SSA array with n types

        Returns:
            - Total number of inactive SSAs across the n types
            - Tuple, beakdown of the number of inactive SSAs for each of the n types
        """
        if ssa_array is None or len(ssa_array) == 0:
            return None
        else:
            total = 0
            num_inactive_per_ssa = []
            for layout in ssa_array:
                n = layout.num_instances - int(np.sum(layout.visible))
                num_inactive_per_ssa.append(n)
                total += n
            return total, tuple(num_inactive_per_ssa)

    def get_visible_ssa_locations(self, ssa_array: List[auto_scene_gen_msgs.StructuralSceneActorLayout]):
        """Get the (x,y) locations for all visible SSAs
        
        Args:
            - ssa_array: Input SSA array

        Returns:
            - List of (2,) numpy arrays indicating the (x,y) locations
        """
        if ssa_array is None or len(ssa_array) == 0:
            return None
        else:
            loc = []
            for layout in ssa_array:
                for i in range(layout.num_instances):
                    if layout.visible[i]:
                        loc.append(np.array([layout.x[i], layout.y[i]], dtype=np.float32))
            return loc

    def add_random_ssa(self, 
                       scenario_request: auto_scene_gen_srvs.RunScenario.Request, 
                       max_attempts: int = None, 
                       ssa_idx: int = None, 
                       center: np.float32 = None, 
                       radius: float = None):
        """Add a single random obstacle to RunScenario request. Can specify a circle within which the obstacle will be placed randomly in. Must specify both center and radius.
        
        Args:
            - scenario_request: RunScenario request to modify
            - max_attempts: The maximum number of attempts, if not None.
            - ssa_idx: (Optional) The SSA index in the SSA setup from which to use. If None, then an index will be randomly chosen.
            - center: (Optional) Specifies the center of a circle from which the obstacle will be placed randomly in
            - radius: (Optional) Specifies the radius of a circle from which to place the obstacle in
        
        Returns:
            - b_success: True if successful, False otherwise
            - attempts: Number of attempts made
            - change_log: The SSA change log
        """
        ssa_array : List[auto_scene_gen_msgs.StructuralSceneActorLayout] = scenario_request.scene_description.ssa_array

        num_inactive, breakdown = self.get_num_inactive_ssa(ssa_array)
        if num_inactive == 0: # All obstacles are active
            return False, 0, None
        
        # Get idxs that we can add obstacles to
        valid_ssa_idx = []
        for i,n in enumerate(breakdown):
            if n > 0:
                valid_ssa_idx.append(i)

        if ssa_idx is None:
            ssa_idx = valid_ssa_idx[np.random.randint(len(valid_ssa_idx))] # Pick the type of SSA now
        else:
            if breakdown[ssa_idx] == 0: # Cannot add any more
                return False, 0, None

        # Use lowest available index for adding the obstacle
        obs_idx = None
        for i in range(ssa_array[ssa_idx].num_instances):
            if not ssa_array[ssa_idx].visible[i]:
                obs_idx = i
                break

        start_location = np.array([scenario_request.vehicle_start_location.x, scenario_request.vehicle_start_location.y])
        goal_location = np.array([scenario_request.vehicle_goal_location.x, scenario_request.vehicle_goal_location.y])

        b_valid = False
        attempts = 0
        request_copy = copy.deepcopy(scenario_request)
        while not b_valid:
            if attempts == max_attempts:
                return False, attempts, None

            ssa_array_copy = copy.deepcopy(ssa_array)
            
            if center is not None and radius is not None:
                angle = np.random.rand() * 2.*math.pi
                distance = np.sqrt(np.random.rand()) * radius
                x = np.clip(center[0] + distance * np.cos(angle), 0., self.landscape_nominal_size)
                y = np.clip(center[1] + distance * np.sin(angle), 0., self.landscape_nominal_size)
            else:
                x = self.ssa_attr.x.random()
                y = self.ssa_attr.y.random()

            yaw = self.ssa_attr.yaw.random()
            scale = self.ssa_attr.scale.random() * self.ssa_config[ssa_idx].max_scale

            # Cannot add obstacle inside obstacle-free radii
            if fast_norm(start_location - np.array([x,y])) <= self.start_obstacle_free_radius:
                attempts += 1
                continue
            if fast_norm(goal_location - np.array([x,y])) <= self.goal_obstacle_free_radius:
                attempts += 1
                continue
            
            ssa_array_copy[ssa_idx].x[obs_idx] = x
            ssa_array_copy[ssa_idx].y[obs_idx] = y
            ssa_array_copy[ssa_idx].yaw[obs_idx] = yaw
            ssa_array_copy[ssa_idx].scale[obs_idx] = scale
            ssa_array_copy[ssa_idx].visible[obs_idx] = True

            request_copy.scene_description.ssa_array = ssa_array_copy
            b_valid = self.is_scenario_request_feasible(request_copy)
            attempts += 1

        # Now we modify the input SSA array
        ssa_array[ssa_idx].x[obs_idx] = x
        ssa_array[ssa_idx].y[obs_idx] = y
        ssa_array[ssa_idx].yaw[obs_idx] = yaw
        ssa_array[ssa_idx].scale[obs_idx] = scale
        ssa_array[ssa_idx].visible[obs_idx] = True

        # Create change log
        change_log = SSAChangeLog()
        change_log.ssa_idx = ssa_idx
        change_log.obs_idx = obs_idx
        change_log.new.x = x
        change_log.new.y = y
        change_log.new.yaw = yaw
        change_log.new.scale = scale
        change_log.new.b_visible = True

        return True, attempts, change_log
    
    def remove_random_ssa(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request, max_attempts: int = None):
        """Remove random obstacle from RunScenario request
        
        Args:
            - scenario_request: RunScenario request to modify
            - max_attempts: The maximum number of attempts, if not None.
        
        Returns:
            - b_success: True if successful, False otherwise
            - attempts: Number of attempts made
            - change_log: The SSA change log
        """
        ssa_array : List[auto_scene_gen_msgs.StructuralSceneActorLayout] = scenario_request.scene_description.ssa_array

        num_active_obs, breakdown = self.get_num_active_ssa(ssa_array)
        if num_active_obs == 0: # Cannot remove what does not exist
            return False, 0, None

        valid_ssa_idx = []
        for i,n in enumerate(breakdown):
            if n > 0:
                valid_ssa_idx.append(i)

        b_valid = False
        attempts = 0
        request_copy = copy.deepcopy(scenario_request)
        while not b_valid:
            if attempts == max_attempts:
                return False, attempts, None

            ssa_array_copy = copy.deepcopy(ssa_array)
            ssa_idx = valid_ssa_idx[np.random.randint(len(valid_ssa_idx))] # Can choose different SSA type each iteration

            # Pick a random active obstacle index
            valid_obs_idx = []
            for i in range(ssa_array_copy[ssa_idx].num_instances):
                if ssa_array_copy[ssa_idx].visible[i]:
                    valid_obs_idx.append(i)
            obs_idx = valid_obs_idx[np.random.randint(len(valid_obs_idx))]

            ssa_array_copy[ssa_idx].visible[obs_idx] = False
            request_copy.scene_description.ssa_array = ssa_array_copy
            b_valid = self.is_scenario_request_feasible(request_copy) # This check should almost always pass, but we still have to verify anyway for consistency purposes
            attempts += 1

        # Update actual SSA array
        ssa_array[ssa_idx].visible[obs_idx] = False

        # Create change log
        change_log = SSAChangeLog()
        change_log.ssa_idx = ssa_idx
        change_log.obs_idx = obs_idx
        change_log.new.x = ssa_array[ssa_idx].x[obs_idx]
        change_log.new.y = ssa_array[ssa_idx].y[obs_idx]
        change_log.new.yaw = ssa_array[ssa_idx].yaw[obs_idx]
        change_log.new.scale = ssa_array[ssa_idx].scale[obs_idx]
        change_log.new.b_visible = False

        return True, attempts, change_log

    def perturb_random_ssa(self, 
                           scenario_request: auto_scene_gen_srvs.RunScenario.Request, 
                           max_attempts: int = None, 
                           ssa_idx: int = None, 
                           center: np.float = None,
                           radius: float = None
                           ):        
        """Perturb a random attribute of a random obstacle in the RunScenario request
        
        Args:
            - scenario_request: RunScenario request to modify
            - max_attempts: The maximum number of attempts, if not None.
            - ssa_idx: (Optional) The SSA index in the SSA setup from which to use. If None, then an index will be randomly chosen.
            - center: (Optional) Specifies the center of a circle from which the obstacle will be placed randomly in
            - radius: (Optional) Specifies the radius of a circle from which to place the obstacle in

        Note: If both the center and radius args are provided, then the x and y attributes will be treated as a a combined attribute called "pos".
        If this new "pos" attribute is chosen, then the new position for the obstacle will be randomly chosen within the specified circle.

        Returns:
            - b_success: True if successful, False otherwise
            - attempts: Number of attempts made
            - change_log: The SSA change log
        """
        ssa_array : List[auto_scene_gen_msgs.StructuralSceneActorLayout] = scenario_request.scene_description.ssa_array

        num_active_obs, breakdown = self.get_num_active_ssa(ssa_array)
        if num_active_obs == 0: # Nothing to perturb
            return False, 0, None

        valid_ssa_idx = []
        for i,n in enumerate(breakdown):
            if n > 0:
                valid_ssa_idx.append(i)
        
        if ssa_idx is None:
            ssa_idx = valid_ssa_idx[np.random.randint(len(valid_ssa_idx))]
        else:
            if breakdown[ssa_idx] == 0: # Nothing to perturb
                return False, 0, None

        # Choose an attribute
        if center is not None and radius is not None: # Combine x and y
            attrs2 = []
            for a in self.ssa_attr.var_attrs:
                if a != self.ssa_attr.X and a != self.ssa_attr.Y:
                    attrs2.append(a)
            if self.ssa_attr.X in self.ssa_attr.var_attrs or self.ssa_attr.Y in self.ssa_attr.var_attrs:
                attrs2.append("pos")
            attr = attrs2[np.random.randint(len(attrs2))]
        else:
            attr = self.ssa_attr.var_attrs[np.random.randint(len(self.ssa_attr.var_attrs))]

        # Pick a random active obstacle index
        valid_obs_idx = []
        for i in range(ssa_array[ssa_idx].num_instances):
            if ssa_array[ssa_idx].visible[i]:
                valid_obs_idx.append(i)
        obs_idx = valid_obs_idx[np.random.randint(len(valid_obs_idx))]
        
        # Get current values
        x = ssa_array[ssa_idx].x[obs_idx]
        y = ssa_array[ssa_idx].y[obs_idx]
        yaw = ssa_array[ssa_idx].yaw[obs_idx]
        scale = ssa_array[ssa_idx].scale[obs_idx]

        # Create change log
        change_log = SSAChangeLog()
        change_log.ssa_idx = ssa_idx
        change_log.obs_idx = obs_idx
        change_log.old.x = x
        change_log.old.y = y
        change_log.old.yaw = yaw
        change_log.old.scale = scale
        change_log.old.b_visible = True

        start_location = np.array([scenario_request.vehicle_start_location.x, scenario_request.vehicle_start_location.y])
        goal_location = np.array([scenario_request.vehicle_goal_location.x, scenario_request.vehicle_goal_location.y])

        b_valid = False
        attempts = 0
        request_copy = copy.deepcopy(scenario_request)
        while not b_valid:
            if attempts == max_attempts:
                return False, attempts, None
            
            ssa_array_copy = copy.deepcopy(ssa_array)

            if attr == "pos":
                angle = np.random.rand() * 2.*math.pi
                distance = np.sqrt(np.random.rand()) * radius
                x = np.clip(center[0] + distance * np.cos(angle), 0., self.landscape_nominal_size)
                y = np.clip(center[1] + distance * np.sin(angle), 0., self.landscape_nominal_size)

            elif attr == self.ssa_attr.X:
                x = self.ssa_attr.x.random()

            elif attr == self.ssa_attr.Y:
                y = self.ssa_attr.y.random()

            elif attr == self.ssa_attr.YAW:
                yaw = self.ssa_attr.yaw.random()
            
            elif attr == self.ssa_attr.SCALE:
                scale = self.ssa_attr.scale.random() * self.ssa_config[ssa_idx].max_scale

            if attr in (self.ssa_attr.X, self.ssa_attr.Y, "pos"):
                # Cannot move obstacle into obstacle-free radii
                if fast_norm(start_location - np.array([x,y])) <= self.start_obstacle_free_radius:
                    attempts += 1
                    continue
                if fast_norm(goal_location - np.array([x,y])) <= self.goal_obstacle_free_radius:
                    attempts += 1
                    continue
            
            ssa_array_copy[ssa_idx].x[obs_idx] = x
            ssa_array_copy[ssa_idx].y[obs_idx] = y
            ssa_array_copy[ssa_idx].yaw[obs_idx] = yaw
            ssa_array_copy[ssa_idx].scale[obs_idx] = scale

            request_copy.scene_description.ssa_array = ssa_array_copy
            b_valid = self.is_scenario_request_feasible(request_copy)
            attempts += 1

        ssa_array[ssa_idx].x[obs_idx] = x
        ssa_array[ssa_idx].y[obs_idx] = y
        ssa_array[ssa_idx].yaw[obs_idx] = yaw
        ssa_array[ssa_idx].scale[obs_idx] = scale

        # Add to change log
        change_log.new.x = x
        change_log.new.y = y
        change_log.new.yaw = yaw
        change_log.new.scale = scale
        change_log.new.b_visible = True

        return True, attempts, change_log
    
    def move_random_ssa(self, 
                        scenario_request: auto_scene_gen_srvs.RunScenario.Request, 
                        max_attempts: int = None, 
                        ssa_idx: int = None,
                        center: np.float32 = None, 
                        radius: float = None):        
        """Move a random SSA to a new random location. Can specify a circle within which the obstacle will be placed randomly in. Must specify both center and radius.
        
        Args:
            - scenario_request: RunScenario request to modify
            - max_attempts: The maximum number of attempts, if not None.
            - ssa_idx: (Optional) The SSA index in the SSA setup from which to use. If None, then an index will be randomly chosen.
            - center: (Optional) Specifies the center of a circle from which the obstacle will be placed randomly in
            - radius: (Optional) Specifies the radius of a circle from which to place the obstacle in

        Returns:
            - b_success: True if successful, False otherwise
            - attempts: Number of attempts made
            - change_log: The SSA change log
        """
        ssa_array : List[auto_scene_gen_msgs.StructuralSceneActorLayout] = scenario_request.scene_description.ssa_array

        num_active_obs, breakdown = self.get_num_active_ssa(ssa_array)
        if num_active_obs == 0: # Nothing to perturb
            return False, 0, None

        valid_ssa_idx = []
        for i,n in enumerate(breakdown):
            if n > 0:
                valid_ssa_idx.append(i)

        if ssa_idx is None:
            ssa_idx = valid_ssa_idx[np.random.randint(len(valid_ssa_idx))]
        else:
            if breakdown[ssa_idx] == 0:
                return False, 0, None

        # Pick a random active obstacle index
        valid_obs_idx = []
        for i in range(ssa_array[ssa_idx].num_instances):
            if ssa_array[ssa_idx].visible[i]:
                valid_obs_idx.append(i)
        obs_idx = valid_obs_idx[np.random.randint(len(valid_obs_idx))]

        # Create change log
        change_log = SSAChangeLog()
        change_log.ssa_idx = ssa_idx
        change_log.obs_idx = obs_idx
        change_log.old.x = ssa_array[ssa_idx].x[obs_idx]
        change_log.old.y = ssa_array[ssa_idx].y[obs_idx]
        change_log.old.yaw = ssa_array[ssa_idx].yaw[obs_idx]
        change_log.old.scale = ssa_array[ssa_idx].scale[obs_idx]
        change_log.old.b_visible = True

        start_location = np.array([scenario_request.vehicle_start_location.x, scenario_request.vehicle_start_location.y])
        goal_location = np.array([scenario_request.vehicle_goal_location.x, scenario_request.vehicle_goal_location.y])

        b_valid = False
        attempts = 0
        request_copy = copy.deepcopy(scenario_request)
        while not b_valid:
            if attempts == max_attempts:
                return False, attempts, None

            ssa_array_copy = copy.deepcopy(ssa_array)

            if center is not None and radius is not None:
                angle = np.random.rand() * 2.*math.pi
                distance = np.sqrt(np.random.rand()) * radius
                x = np.clip(center[0] + distance * np.cos(angle), 0., self.landscape_nominal_size)
                y = np.clip(center[1] + distance * np.sin(angle), 0., self.landscape_nominal_size)
            else:
                x = self.ssa_attr.x.random()
                y = self.ssa_attr.y.random()

            # Cannot add obstacle inside obstacle-free radii
            if fast_norm(start_location - np.array([x,y])) <= self.start_obstacle_free_radius:
                attempts += 1
                continue
            if fast_norm(goal_location - np.array([x,y])) <= self.goal_obstacle_free_radius:
                attempts += 1
                continue
            
            ssa_array_copy[ssa_idx].x[obs_idx] = x
            ssa_array_copy[ssa_idx].y[obs_idx] = y

            request_copy.scene_description.ssa_array = ssa_array_copy
            b_valid = self.is_scenario_request_feasible(request_copy)
            attempts += 1

        ssa_array[ssa_idx].x[obs_idx] = x
        ssa_array[ssa_idx].y[obs_idx] = y

        # Add to change log
        change_log.new.x = x
        change_log.new.y = y
        change_log.new.yaw = ssa_array[ssa_idx].yaw[obs_idx]
        change_log.new.scale = ssa_array[ssa_idx].scale[obs_idx]
        change_log.new.b_visible = True

        return True, attempts, change_log

    def perturb_rand_txt_attr(self, scenario_request: auto_scene_gen_srvs.RunScenario.Request):
        """Perturb a random textural attribute in the given scenario request
        
        Args:
            - scenario_request: RunScenario request to modify

        Returns:
            True if successful, False otherwise
        """
        if len(self.txt_attr.var_attrs) == 0:
            return False
        
        attr = self.txt_attr.var_attrs[np.random.randint(len(self.txt_attr.var_attrs))]

        if attr == self.txt_attr.SUNLIGHT_INCLINATION:
            scenario_request.scene_description.sunlight_inclination = self.txt_attr.sunlight_inclination.random()
            return True

        elif attr == self.txt_attr.SUNLIGHT_YAW:
            scenario_request.scene_description.sunlight_yaw_angle = self.txt_attr.sunlight_yaw.random()
            return True