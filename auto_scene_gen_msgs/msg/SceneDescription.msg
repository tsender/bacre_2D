# This message contains information regarding the scene description

auto_scene_gen_msgs/LandscapeDescription landscape # Landscape description

float32 sunlight_inclination # The angle the sunlight makes with the horizontal [deg]
float32 sunlight_yaw_angle # The yaw angle the sunlight is pointing in (i.e., the angle the shadow will be cast in) [deg]

auto_scene_gen_msgs/StructuralSceneActorLayout[] ssa_array   # An array defining all of the types of SSAs and their attributes from which to place in the UE4 scene