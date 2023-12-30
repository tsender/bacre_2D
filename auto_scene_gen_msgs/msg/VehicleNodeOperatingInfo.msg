# This message is published by the AutoSceneGenClient to tell all registered vehicle nodes important operating information 
# This info includes: if all vehicle nodes for that worker are ready, where the nodes should save their internal data and how much data to save
# Each array has the same length and all data is organized according to the order of the worker ID in the 'worker_ids' field.

uint8[] worker_ids          # AutoSceneGen worker IDs
uint32[] scenario_numbers   # List of the scenario numbers being run on each AutoSceneGen worker, listed in the same order as worker_ids
bool[] ok_to_run            # Indicates if it is okay for the vehicle nodes to run, listed in the same order as worker_ids
string[] save_dirs          # List of directories to save data to, listed in the same order as worker_ids
bool[] save_minimal         # List of booleans to indicate if the vehicle node should only save the least amount of data required (and skip any time-consuming analyses),
                            # listed in the same order as worker_ids