# Message for sending basic PhysX control commands to a PhysX vehicle

std_msgs/Header header
float32 longitudinal_velocity   # Vehicle's longitudinal velocity in [cm/s or m/s]
float32 steering_angle          # Range [-MaxSteeringAngle, +MaxSteeringAngle] [deg]
bool handbrake                  # false = disengaged, true = engaged