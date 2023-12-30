# Message for specifying scene capture settings. The camera used has a 60 deg. FOV.

uint32 image_size                       # The image size in pixels. All images will be square.

# Annotations to add to the scene captures
bool draw_annotations                   # Indicates if the scene captures should contain annotations
float32 goal_sphere_thickness           # Goal sphere thickness in [cm]
std_msgs/ColorRGBA goal_sphere_color    # Goal sphere color

# Bird's-eye / Aerial Views
bool ortho_aerial                       # Draw orthographic aerial view
bool perspective_aerial                 # Draw perspective aerial view
uint32[] aerial_padding                 # Padding in [m] to apply to the base orthographic width

# Perspective aerial-like views of the various sides/corners of the landscape. All views are from the outside looking in.
bool front_aerial               # Draw perspective aerial-like view from the front
bool left_front_aerial          # Draw perspective aerial-like view from the left-front
bool left_aerial                # Draw perspective aerial-like view from the left
bool left_rear_aerial           # Draw perspective aerial-like view from the left-rear
bool rear_aerial                # Draw perspective aerial-like view from the rear
bool right_rear_aerial          # Draw perspective aerial-like view from the right rear
bool right_aerial               # Draw perspective aerial-like view from the right
bool right_front_aerial         # Draw perspective aerial-like view from the right-front

bool vehicle_start_pov           # Draw perspective view of the vehicle's POV at its starting location
bool vehicle_start_rear_aerial   # Draw perspective 3rd person rear aerial view of the vehicle at the starting location

# TODO: Allow the user to specify custom poses from which to take pictures from