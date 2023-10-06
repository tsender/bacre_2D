import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback
import threading
from typing import Tuple, List

def fast_norm(vector: np.float32, axis: int = None):
    """Faster way to compute numpy vector norm than np.linalg.norm()"""
    return np.sqrt(np.sum(np.square(vector), axis=axis))

class SearchState:
    def __init__(self, uid: np.uint16, proj_cost: float, cost_incurred: float, path: List[Tuple[np.uint16]]):
        self.uid = uid
        self.proj_cost = proj_cost
        self.cost_incurred = cost_incurred
        self.path = path

    def __lt__(self, other):
        """Override < comparison operator"""
        return self.proj_cost < other.proj_cost

    def __le__(self, other):
        """Override <= comparison operator"""
        return self.proj_cost <= other.proj_cost

    def __gt__(self, other):
        """Override > comparison operator"""
        return self.proj_cost > other.proj_cost

    def __ge__(self, other):
        """Override >= comparison operator"""
        return self.proj_cost >= other.proj_cost
    
    def __eq__(self, other):
        """Override == comparison operator"""
        return self.proj_cost == other.proj_cost

    def __ne__(self, other):
        """Override != comparison operator"""
        return self.proj_cost != other.proj_cost


class AstarVotingPathPlanner:
    """
    This path planner discretizes a continuous landscape into a grid in which the corners of each grid cell are the vertices that may be traversed.
    To make things easier, in Euclidean space the origin is at the lower left corner of the (nominal) grid and so we will be using a (col, row) = (x_idx, y_idx) format for idxs with row 0 at the bottom.
    Effectively cols and rows correspond to tick marks on the XY axes.
    """
    def __init__(self, 
                env_size: Tuple[float],         # The size of the environment (X,Y) in [m]
                border_size: float,             # A border that extends in all four Cartesianel directions equally, Should be divisible by minor and major cell sizes
                minor_cell_size: float,         # The size of each minor grid cell in [m] (determines the cells that obstacles can be placed into for binning)
                major_cell_size: float,         # The size of each major grid cell in [m] (determines the vertices the path planner can consider)
                bell_curve_sigma: float,        # Standard deviation for the bell curve in [m]
                bell_curve_alpha: float,        # Bell curve scale factor
                max_votes_per_vertex: int,      # Maximum number of votes allowed for each minor vertex (used to indicate if an obstacle is near this point)
                radius_threashold1: float,      # 
                radius_threashold2: float,      # 
                use_three_point_edge_cost: bool = True,   # Indicates if the edge cost should use the two endpoints and the midpoint. False means only use the midpoint (faster calculation, but less accurate).
                ):
        
        self.env_size = np.array(env_size)
        self.border_size = border_size

        # Minor / obstacle grid vertices (Does NOT include the border)
        num_x_vertices = int(round(env_size[0] / minor_cell_size)) + 1
        num_y_vertices = int(round(env_size[1] / minor_cell_size)) + 1
        self.minor_grid_shape = (num_x_vertices, num_y_vertices) # (num vertices along x-axis, num vertices along y-axis)
        self.minor_cell_size = np.array([minor_cell_size, minor_cell_size])
        self.num_minor_vertices = num_x_vertices * num_y_vertices

        # Create a 2D array of the (x,y) locations for each minor vertex
        x_grid, y_grid = np.meshgrid(np.arange(num_x_vertices, dtype=np.float32) / (num_x_vertices-1), np.arange(num_y_vertices, dtype=np.float32) / (num_y_vertices-1))
        x_grid = x_grid.reshape(num_y_vertices, num_x_vertices, 1)
        y_grid = y_grid.reshape(num_y_vertices, num_x_vertices, 1)
        self.minor_vertices = np.concatenate([x_grid, y_grid], axis=-1).reshape(self.num_minor_vertices, 2) # The i-th row is the normalized (x,y) location for the minor vertex with uid = i
        self.minor_vertices[:,0] *= self.env_size[0]
        self.minor_vertices[:,1] *= self.env_size[1]

        # Major / vehicle grid vertices
        num_x_vertices = int(round((env_size[0] + 2.*border_size) / major_cell_size)) + 1
        num_y_vertices = int(round((env_size[1] + 2.*border_size) / major_cell_size)) + 1
        self.major_grid_shape = (num_x_vertices, num_y_vertices) # (num vertices along x-axis, num vertices along y-axis)
        self.major_cell_size = np.array([major_cell_size, major_cell_size], dtype=np.float32)

        # Bell curve
        self.bell_curve_sigma = bell_curve_sigma
        self.bell_curve_alpha = bell_curve_alpha
        self.bell_curve_covariance = (self.bell_curve_sigma**2) * np.identity(2, dtype=np.float32)
        self.inv_bell_curve_covariance = np.linalg.inv(self.bell_curve_covariance)

        self.radius_threashold1 = radius_threashold1
        self.radius_threashold2 = radius_threashold2
        self.use_three_point_edge_cost = use_three_point_edge_cost

        # Obstacle stuff
        self.max_votes_per_vertex = max_votes_per_vertex
        self.obstacle_votes = np.zeros(self.num_minor_vertices)
        self.obs_locations_compact = {}
        self.obs_locations_compact_np = None
        self.obs_votes_compact = {}
        self.obs_votes_compact_np = None

    # def get_unnormalized_minor_vertices(self):
    #     vertices = self.minor_vertices.copy()
    #     vertices[:,0] *= (self.env_size[0] + 2.*self.border_size)
    #     vertices[:,1] *= (self.env_size[1] + 2.*self.border_size)
    #     return vertices - self.border_size
    
    # def get_unnormalized_vertex(self, vertex: np.float32):
    #     factor = np.array([self.env_size[0] + 2.*self.border_size,self.env_size[1] + 2.*self.border_size])
    #     return (vertex * factor) - self.border_size

    def xy_to_idx(self, xy: np.float32, b_major: bool = True):
        """Converts an (x,y) position in [m] to the appropriate idx in (x_idx, y_idx) format
        
        Args:
            - xy: The input (x,y) position in [m] as a numpy array

        Returns:
            The corresponding (x_idx, y_idx)
        """
        if b_major:
            p = xy + self.border_size
            return tuple(np.round(p / self.major_cell_size))
        else:
            p = xy
            return tuple(np.round(p / self.minor_cell_size))
        
    def xy_to_uid(self, xy: np.float32, b_major: bool = True):
        """Converts an (x,y) position in [m] to the appropriate UID
        
        Args:
            - xy: The input (x,y) position in [m] as a numpy array

        Returns:
            The corresponding UID
        """
        idx = self.xy_to_idx(xy, b_major)
        return self.idx_to_uid(idx, b_major)

    def idx_to_xy(self, idx: Tuple[np.uint16], b_major: bool = True):
        """Converts (x_idx, y_idx) idx to corresponding (x,y) position as numpy array
        
        Args:
            - idx: The input (x_idx, y_idx)

        Returns:
            The corresponding (x,y) position in [m] as numpy array
        """
        if b_major:
            return idx * self.major_cell_size - self.border_size
        else:
            return idx * self.minor_cell_size #- self.border_size


    def idx_to_uid(self, idx: Tuple[np.uint16], b_major: bool = True):
        """Get universal ID (UID) from idx. Lower-left is 0, uid increases to the right and up.
        
        Args:
            - idx: Current index as a tuple (x_idx, y_idx)
        
        Returns:
            UID of indexed grid cell
        """
        x_idx, y_idx = idx
        if b_major:
            return int(self.major_grid_shape[0] * y_idx + x_idx)
        else:
            return int(self.minor_grid_shape[0] * y_idx + x_idx)


    def uid_to_idx(self, uid: np.uint16, b_major: bool = True):
        """Get idx from uid
        
        Args:
            uid: UID of a grid cell
        Returns:
            Index as a tuple (x_idx, y_idx)
        """
        if b_major:
            y_idx = int(uid//self.major_grid_shape[0])
            x_idx = int(uid - y_idx * self.major_grid_shape[0])
        else:
            y_idx = int(uid//self.minor_grid_shape[0])
            x_idx = int(uid - y_idx * self.minor_grid_shape[0])

        return (x_idx, y_idx)

    def uid_to_xy(self, uid: np.uint16, b_major: bool = True):
        """Get (x,y) in [m] from uid
        
        Args:
            uid: UID of a grid cell
        Returns:
            The corresponding (x,y) position in [m] as numpy array
        """
        idx = self.uid_to_idx(uid, b_major)
        return self.idx_to_xy(idx, b_major)

    def is_idx_valid(self, idx: Tuple[np.uint16], b_major: bool = True):
        """Indicates if idx is valid (exists in the grid)
        
        Args:
            idx: current index as a tuple (x_idx, y_idx)
            
        Returns:
            True if idx is valid, False otherwise
        """
        if b_major and 0 <= idx[0] < self.major_grid_shape[0] and 0 <= idx[1] < self.major_grid_shape[1]:
            return True
        if not b_major and 0 <= idx[0] < self.minor_grid_shape[0] and 0 <= idx[1] < self.minor_grid_shape[1]:
            return True
        return False

    def get_valid_neighbors(self, idx):
        """Find valid neighborsin major grid
        
        Args:
            idx: current index as a tuple (x_idx, y_idx)
            
        Returns:
            A list of valid/open neighbor idx's
        """
        # Note: Including diagonal neighbors significantly reduces computation time
        neighbors = [(idx[0], idx[1] + 1), # Up
                      (idx[0], idx[1] - 1), # Down
                      (idx[0] + 1, idx[1]), # Right
                      (idx[0] - 1, idx[1]), # Left
                      (idx[0] + 1, idx[1] + 1), # UR
                      (idx[0] - 1, idx[1] + 1), # UL
                      (idx[0] + 1, idx[1] - 1), # LR
                      (idx[0] - 1, idx[1] - 1)] # LL

        valid = []
        for n_idx in neighbors:
            if self.is_idx_valid(n_idx):
                valid.append(n_idx)
        return valid

    def get_nearest_vertex(self, location: np.float32, b_major: bool = True):
        """Find the nearest vertex to the specified location
        
        Args:
            - location: The specified location
            - b_major: Indicates if we want the nearest major vertex (False for minor vertex)

        Returns:
            The the nearest vertex in [m] to the specified location
        """
        if b_major:
            loc = np.minimum(np.maximum([-self.border_size, -self.border_size], location), self.env_size + self.border_size) # Keep location within defined environment bounds
        else:
            loc = np.minimum(np.maximum([0., 0.], location), self.env_size) # Keep location within defined environment bounds
        nearest_idx = self.xy_to_idx(loc, b_major)
        return self.idx_to_xy(nearest_idx, b_major)

    def clear_map(self):
        """Call these when you want to clear all obstacle locations and votes in the map"""
        self.obstacle_votes[:] = 0.
        self.obs_locations_compact.clear()
        self.obs_votes_compact.clear()

    def add_obstacle_vote(self, location = np.float32, votes_to_add: int = 1):
        """Add obstacle vote to the nearest minor vertex to the given location
        
        Args:
            - location: The actual obstacle (x,y) location
            - votes_to_add: Number of votes to add at this location
        
        Returns:
            - True if able to add obstacle, false otherwise
        """
        if 0 <= location[0] <= self.env_size[0] and 0 <= location[1] <= self.env_size[1]: # Can not add obstacles in bordered region
            # if b_snap_to_minor_grid:
            nearest_vertex = self.get_nearest_vertex(location, b_major=False)
            uid = self.xy_to_uid(nearest_vertex, b_major=False)
            self.obstacle_votes[uid] = min(self.obstacle_votes[uid] + votes_to_add, self.max_votes_per_vertex)

            # Add to compact arrays
            self.obs_votes_compact[uid] = float(self.obstacle_votes[uid])
            if uid not in self.obs_locations_compact.keys():
                self.obs_locations_compact[uid] = nearest_vertex
                # if uid in self.obs_locations_compact.keys():
                #     self.obs_votes_compact[uid] = float(min(self.obs_votes_compact[uid] + votes_to_add, self.max_votes_per_vertex))
                # else:
                #     self.obs_locations_compact[uid] = nearest_vertex
                #     self.obs_votes_compact[uid] = float(votes_to_add)
            # else:
            #     # Each call to this function adds a new obstacle, regardless if it coincides with an existing one
            #     uid = len(self.obs_locations_compact)
            #     self.obs_locations_compact[uid] = location
            #     self.obs_votes_compact[uid] = votes_to_add
            return True
        return False
    
    def add_obstacle_votes_from_array(self, locations: np.float32):
        """Vectorized version of add_obstacle_vote(), and allows multiple locations to be passed in a single array. Very fast/efficient for large arrays.

        Args:
            - locations: (n,2) numpy array of xy locations in [m] for every detected obstacle. Input locations can be in continuous space,
                        as this function will determine the closest minor vertex.
        """
        valid_idxs = (locations[:,0] >= 0.) & (locations[:,1] >= 0.) & (locations[:,0] <= self.env_size[0]) & (locations[:,1] <= self.env_size[1])
        if np.count_nonzero(valid_idxs):
            actual_xy = locations[valid_idxs,:]
            nearest_minor_idx = np.round(actual_xy / self.minor_cell_size)
            nearest_minor_xy = nearest_minor_idx * self.minor_cell_size
            nearest_minor_uid = np.array(self.minor_grid_shape[0] * nearest_minor_idx[:,1] + nearest_minor_idx[:,0], dtype=np.uint32)

            sorted_uids, first_idx, uid_counts = np.unique(nearest_minor_uid, return_index=True, return_counts=True)
            for i in range(sorted_uids.size):
                uid = sorted_uids[i]
                xy = nearest_minor_xy[first_idx[i]]
                votes_to_add = uid_counts[i]

                self.obstacle_votes[uid] = min(self.obstacle_votes[uid] + votes_to_add, self.max_votes_per_vertex)
                self.obs_votes_compact[uid] = float(self.obstacle_votes[uid])
                if uid not in self.obs_locations_compact.keys():
                    self.obs_locations_compact[uid] = xy

        return np.count_nonzero(valid_idxs)

    def update_compact_numpy_arrays(self):
        self.obs_locations_compact_np = np.array(list(self.obs_locations_compact.values())) # Each row is a (x,y) location
        self.obs_votes_compact_np = np.array(list(self.obs_votes_compact.values())) # 1-D array

    def get_arbitrary_path_cost(self, path: List[np.float32]):
        """Compute the path cost for an arbitrary path (points need not lie on major vertices).
        This is perfectly okay because the path has a finite number of points in it.
        
        Args:
            - path: List of ordered points with each element an array of (x,y). Can also be numpy array of size (n,2)

        Returns:
            - Normalized path cost
        """
        cost = 0.
        for i in range(len(path)-1):
            cost += self.edge_travel_cost(path[i], path[i+1])
        return cost
    
    def edge_travel_cost(self, p1: np.float32, p2: np.float32, start_point: np.float32 = None, start_yaw: float = None):
        """Compute the edge travel cost from point p1 to point p2
        Travel Cost = 2D Euclidena distance * (1 + Gaussian pdf scaling from the obstacles)

        Args:
            - p1: The first (x,y) point of the edge in [m]
            - p2: The second (x,y) point of the edge in [m]
            - start_point: (Optional) The starting point in (x,y) [m] for the entire path. Only used if start_yaw is also provided.
            - start_yaw: (Optional) Indicates a starting yaw angle in [rad] for the agent following this path. See construct_optimal_path()

        Returns:
            The travel cost
        """
        mid = (p1 + p2) / 2.
        euclid_dist = fast_norm(p1 - p2)
        
        # Only account for existing obstacle locations (where votes > 0). 
        # Vectorized bell curve calculations (MUCH faster than simple for-loop approach)
        # Also MUCH faster than assuming an obstacle at each possible location
        if self.obs_locations_compact_np.shape[0] > 0:
            if self.use_three_point_edge_cost:
                refs = [p1, mid, p2] # Use the two endpoints and midpoint as reference points. Then we'll take the average at the end
            else:
                refs = [mid]
            deltas = [p-self.obs_locations_compact_np for p in refs] # Each row corresponds to (x-mean).T. This is an (N,2) array
            quad_terms = [np.sum(delta.dot(self.inv_bell_curve_covariance) * delta, axis=1) for delta in deltas] # 1-D array, each element is (x-mean).T.dot(cov^-1).dot(x-mean)
            scale_factors = [np.sum(self.obs_votes_compact_np * self.bell_curve_alpha * np.exp(-0.5 * quad_term)) for quad_term in quad_terms]
            scale_factor = np.mean(scale_factors)

            # delta = mid - self.obs_locations_compact_np # Each row corresponds to (x-mean).T. This is an (N,2) array
            # quad_term = np.sum(delta.dot(self.inv_bell_curve_covariance) * delta, axis=1) # 1-D array, each element is (x-mean).T.dot(cov^-1).dot(x-mean)
            # scale_factor = np.sum(self.obs_votes_compact_np * self.bell_curve_alpha * np.exp(-0.5 * quad_term))
        else:
            scale_factor = 0.

        # Using einsum (actually a bit slower)
        # quad = np.einsum("ji,jk,ki->i", delta.T, self.inv_bell_curve_covariance, delta.T)
        # quad = np.einsum("ij,jk,ik->i", delta, self.inv_bell_curve_covariance, delta)
        # scale_factor = self.bell_curve_alpha * np.einsum("i,i->", self.obstacle_votes, np.exp(-0.5*quad))

        directional_cost = 0.
        if start_point is not None and start_yaw is not None and fast_norm(p2 - p1) > 1e-5 and fast_norm(p2 - start_point) <= self.radius_threashold2:
            v = np.array([np.cos(start_yaw), np.sin(start_yaw)], dtype=np.float32) # Unit vector in direction of start_yaw
            u12 = (p2 - p1) / fast_norm(p2 - p1)
            theta = np.arccos(np.clip(v.dot(u12), -1., 1.))

            # In case the agent goes out of bounds, the agent will need to create a path that "appears" to move backwards, hence we cannot use infinity here
            if fast_norm(p2 - start_point) <= self.radius_threashold1 and theta > 45.*math.pi/180:
                directional_cost = 1e6

            if self.radius_threashold1 < fast_norm(p2 - start_point) <= self.radius_threashold2 and theta > 72.5*math.pi/180:
                directional_cost = 1e6

        return euclid_dist * (1. + scale_factor) + directional_cost

    def get_edge_cost_breakdown(self, path: List[np.float32], start_yaw: float = None):
        """Get the cost of each edge in a path (can be arbitrary) and the total cost
        
        Args:
            - path: List of (x,y) points on a path
            - start_yaw: (Optional) Indicates a starting yaw angle in [rad] for the agent following this path. See construct_optimal_path() for more details.
        
        Returns:
            - edge_cost: List of costs for each edge (in order)
            - total_cost: Total path cost
        """
        total_cost = 0.
        edge_cost = []
        for i in range(len(path)-1):
            cost = self.edge_travel_cost(path[i], path[i+1], path[0], start_yaw)
            total_cost += cost
            edge_cost.append(cost)
        return edge_cost, total_cost

    def astar_heuristic(self, pos: np.float32, goal_point: np.float32):
        """Calculates the heuristic for A*, which is the Euclidean distance
        Args:
            - pos: Current point as (x,y)
            - goal_point: The goal point as (x,y)
        
        Returns:
            The heuristic based on the current location
        """
        return fast_norm(pos - goal_point)

    def construct_optimal_path(self, start_point: np.float32, goal_point: np.float32, start_yaw: float = None, event: threading.Event = None):
        """Find an optimal path using A*.

        ARgs:
            - start_point: The starting (x,y) point in [m] that lies on the major grid
            - goal_point: The goal (x,y) point in [m] that lies on the major grid
            - start_yaw: (Optional) Indicates a starting yaw angle in [rad] for the agent following this path. If provided, we will use the radius threshlds to ensure the
                        initial path segments do not require "large" yaw angle changes.
        
        Returns:
            - opt_path: The optimal path as a list of ordered numpy arrays from start_point to goal_point
            - opt_cost: The cost of the optimal path
            - iterations: The number of A* iterations performed
        """
        open_set : List[SearchState] = [] # Priority Queue
        closed_set = [] # List of uid's visited
        start_idx = self.xy_to_idx(start_point)
        goal_idx = self.xy_to_idx(goal_point)
        ss = SearchState(self.idx_to_uid(start_idx), self.astar_heuristic(start_point, goal_point), 0., [start_idx])
        heapq.heappush(open_set, ss)

        idx = start_idx
        at_goal_state = False
        iterations = 0

        while idx != goal_idx and len(open_set) != 0:
            iterations = iterations + 1
            ss = heapq.heappop(open_set)
            idx = self.uid_to_idx(ss.uid)

            if event is not None and event.is_set():
                return None, None, iterations

            if idx == goal_idx:
                at_goal_state = True
                break

            if ss.uid not in closed_set:
                closed_set.append(ss.uid)
                
                for n_idx in self.get_valid_neighbors(idx):
                    n_uid = self.idx_to_uid(n_idx)
                    if n_uid not in closed_set: # Unvisited, open tile
                        p1 = self.idx_to_xy(idx)
                        p2 = self.idx_to_xy(n_idx)

                        n_cost_incurred = ss.cost_incurred + self.edge_travel_cost(p1, p2, start_point, start_yaw)
                        n_proj_cost = n_cost_incurred + self.astar_heuristic(p2, goal_point) #- self.astar_heuristic(idx)
                        n_path = ss.path.copy()
                        n_path.append(n_idx)
                        new_ss = SearchState(n_uid, n_proj_cost, n_cost_incurred, n_path)
                        heapq.heappush(open_set, new_ss)

        if at_goal_state:
            opt_path = []
            for idx in ss.path:
                opt_path.append(self.idx_to_xy(idx))
            return opt_path, ss.cost_incurred, iterations
        else:
            return None, None, iterations

    def compute_path_length(self, waypoints: List[np.float32]):
        """Calculates the path arc length

        Args:
            - waypoints: List of ordered waypoints to pass through
        
        Returns:
            Path arc length
        """
        length = 0.
        for i in range(len(waypoints)-1):
            length += fast_norm(waypoints[i] - waypoints[i+1])
        return length
    
    def create_valid_path_from_waypoints(self, waypoints: List[np.float32]):
        """Create a valid path (follows edges of the major grid) to pass as closely to the requested waypoints.

        Args:
            - waypoints: List of ordered waypoints to pass through/near
        
        Returns:
            Path waypoints as list, path cost
        """
        start_point = self.get_nearest_vertex(waypoints[0])
        start_idx = self.xy_to_idx(start_point)

        path_waypoints = [start_point]
        path_idxs = [start_idx]
        path_cost = 0.

        for i in range(len(waypoints)-1):
            next_point = self.get_nearest_vertex(waypoints[i+1])
            next_idx = self.xy_to_idx(next_point)
                
            while path_idxs[-1] != next_idx:
                min_dist = math.inf
                closest_idx = None
                for n_idx in self.get_valid_neighbors(path_idxs[-1]):
                    n_xy = self.idx_to_xy(n_idx)
                    dist = fast_norm(n_xy - next_point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = n_idx

                closest_point = self.idx_to_xy(closest_idx)
                edge_cost = self.edge_travel_cost(path_waypoints[-1], closest_point)
                path_cost += edge_cost
                path_waypoints.append(closest_point)
                path_idxs.append(closest_idx)

        return path_waypoints, path_cost

    def distribute_path_waypoints(self, path_vertices: List[np.float32], num_path_waypoints: int):
        """Takes a list of path waypoints and creates a modified path in which all waypoints are evenly distributed 
        across the arc length of the path
        
        Args:
            - path_vertices: The original path vertices (the true path)
            - num_path_waypoints: Number of points that the new path should have

        Returns:
            A new path with the specified number of waypoints evenly distributed across the arc length of the original path
        """
        # Compute segment and cumulative segment distances
        segment_distances = []
        cumulative_distances = [0]
        for i in range(len(path_vertices)-1):
            segment_distances.append(fast_norm(path_vertices[i] - path_vertices[i+1]))
            cumulative_distances.append(np.sum(segment_distances))
        
        # Discretize the path into n waypoints
        step_size = np.sum(segment_distances) / (num_path_waypoints-1)
        distributed_waypoints = []
        seg = 0 # Number of segs fully traveled
        for i in range(num_path_waypoints):
            r = (i * step_size - cumulative_distances[seg]) / segment_distances[seg] # Ratio of where the current point is along the current segment
            while r > 1. + 1e-8:
                seg += 1
                r = (i * step_size - cumulative_distances[seg]) / segment_distances[seg]
            p = path_vertices[seg] + r * (path_vertices[seg+1] - path_vertices[seg])
            distributed_waypoints.append(p)
        
        return distributed_waypoints

if __name__ == "__main__":
    import time
    max_votes = 50
    planner = AstarVotingPathPlanner((60.,60.), 100., 1., 2., 1.5, 0.1, max_votes, 5., 10.)
    obs_vertices = planner.minor_vertices * planner.env_size

    # num_obs = 100 # For a small number of obstacles (<50), the major grid can be of size 200x200 vertices
    # print(f"Num obstacles = {num_obs}")
    # add_start = time.time()
    # for _ in range(num_obs):
    #     i = np.random.randint(obs_vertices.shape[0])
    #     planner.add_obstacle_vote(obs_vertices[i], votes_to_add=50)
    # print(f"added obs in {time.time() - add_start} seconds")

    planner.add_obstacle_vote(np.array([30.,  30.]), votes_to_add=max_votes)
    planner.add_obstacle_vote(np.array([20.,  11.]), votes_to_add=max_votes)
    planner.add_obstacle_vote(np.array([19.,  12.]), votes_to_add=50)
    planner.update_compact_numpy_arrays()

    start = np.array([10.,10.])
    goal = np.array([50.,50.])
    ref_start = time.time()
    ref_path, ref_path_cost = planner.create_valid_path_from_waypoints([start, (start+goal)/2, goal])
    ref_cost_breakdown, ref_cost2 = planner.get_edge_cost_breakdown(ref_path)
    ref_cost_breakdown.insert(0, 0.)
    print(f"REF done in {time.time()-ref_start} seconds")
    print(f"ref path cost: {ref_path_cost}")
    for p,c in zip(ref_path,ref_cost_breakdown):
        print(f"ref path = {p}, cost = {c}")
    
    starting_yaw = 45. * math.pi/180.
    print("")
    opt_start = time.time()
    opt_path, opt_path_cost, iterations = planner.construct_optimal_path(start, goal)
    print(f"OPT done in {time.time()-opt_start} seconds")
    print(f"opt path cost: {opt_path_cost}")
    print(f"opt path iterations: {iterations}")
    for p in opt_path:
        print(f"opt path = {p}")