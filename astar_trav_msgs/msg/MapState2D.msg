# This message indicates where predicted obstacles lie in a 2D discretized map and how many votes there are at each obstacle location.

std_msgs/Header header
float32 env_size_x
float32 env_size_y
float32 border_size
float32 minor_cell_size
float32 bell_curve_sigma

float32[] obstacle_votes
float32[] obstacle_votes_compact         # Array of size n, where n is the number of detected obstacle locations
float32[] obstacle_locations_compact_x
float32[] obstacle_locations_compact_y