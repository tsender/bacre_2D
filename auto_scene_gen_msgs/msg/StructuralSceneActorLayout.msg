# This message specifies all of the attributes for the actors of a specific SSA subclass that we wish to place in a UE4 scene

string path_name    # UE path name for the SSA subclass

uint16 num_instances # For verifications purposes, this is used to determine the number of instances

bool[] visible             # Indicates if the actor is visible in the game. 
                            # Toggling this is more efficient than adding/removing elements when creating the UE4 scene
                
bool[] cast_shadow         # Indicates if the actor can cast a shadow in the game

float32[] x                 # X coordinates in [cm or m]
float32[] y                 # Y coordinates in [cm or m]
float32[] yaw               # Yaw angles in [deg]
float32[] scale             # The mesh's scale (applies to all 3 axes)