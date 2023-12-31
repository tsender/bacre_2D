# Black-Box Adversarially Compounding Regret Through Evolution (BACRE) Applied to a 2D Off-Road Autonomy System

## Description

This codebase provides an implementation of BACRE as applied to off-road autonomus vehicles (AVs) using the [AutomaticSceneGeneration](https://github.com/tsender/AutomaticSceneGeneration) plugin for UE4. While the implementation of BACRE can be extended to 3D, the AutomaticSceneGeneration currently only provides support for creating 2D off-road scenes (hence the "2D" in the title).

This codebase contains the following ROS packages:
1. `adversarial_scene_gen`: Contains the primary ROS node for BACRE.
2. `astar_trav`: Contains the code needed to run the A*-traversability autonomy system as used in the experiments in the BACRE paper.
3. `astar_trav_msgs`: Contains custom ROS messages for the `astar_trav` package.
4. `auto_scene_gen_core`: Contains the main ROS nodes and objects needed to interact with an AutoSceneGenWorker and AutoSceneGenVehicle in UE4.
5. `auto_scene_gen_msgs`: Contains the custom message and service definitions for the AutoSceneGen platform.

Note: For convenience, both `auto_scene_gen_*` packages have been copied directly from the [auto_scene_gen](https://github.com/tsender/auto_scene_gen) ROS2 interface to keep all of the code in a single repository.

## Citation

If you use our work in an academic context, we would greatly appreciate it if you used the following citation:

TODO

## Running the Experiments from the Paper

### Installation

Please refer to the [AutomaticSceneGeneration](https://github.com/tsender/AutomaticSceneGeneration) plugin for UE4 and the corresponding ROS2 interface [auto_scene_gen](https://github.com/tsender/auto_scene_gen) for the software requirements to use this codebase. All of our UE4 simulations were ran on computers running Windows 10 with UE 4.26, and all of the ROS code was developed/executed in docker containers, with a Ubuntu 20.04 docker image, running on linux machines. You can download our docker image with the tag `tsender/tensorflow:gpu-focal-foxy` (you may need to login to your docker account from the command line to pull the image).

### UE4 Setup Part 1: Requirements

In UE4 we need to create an AutoSceneGenVehicle actor, create a few StructuralSceneActors, setup the AutoSceneGenWorker, and then configure rosbridge. It is assumed that the units in Unreal Engine is configured for centimeters for position and degrees for orientation (these should be the default settings).

#### AutoSceneGenVehicle

The experiments in the paper use a Polaris MRZR virtual vehicle model (if you do not have access to such a model, then any other car model will suffice for learning how to use the code). Follow these steps to create an [AutoSceneGenVehicle](https://github.com/tsender/AutomaticSceneGeneration/blob/main/Documentation/actors.md) child blueprint, and use the appropriate skeletal mesh for the vehicle model you have at hand. Under the Details panel for this actor, we will use the default values. Under the `Auto Scene Gen Vehicle` tab, make sure the following parameters are set to:
- `Linear Motion Threshold`: 2
- `Vehicle Name`: "vehicle"

Now we need to attach the [sensors](https://github.com/tsender/AutomaticSceneGeneration/blob/main/Documentation/sensors.md). The vehicle in the experiments has a forward-facing camera and a localization sensor.
- Open up your vehicle blueprint
- Add the camera
    - Add a CompleteCameraSensor at the location (150, 0, 140) centimeters relative to the mesh origin, with a zero roll, pitch, and yaw. Note, these coordinates are in the left-hand coordinate system.
    - Set both the `Image Height` and `Image Width` to 256.
    - Set the `Frame Rate` to 15 Hz.
    - Check the `Enable Depth Cam` box to enable the depth camera.
    - Set the `Sensor Name` to "camera" (the default value).
- Add the locaization sensor
    - Add a LocalizationSensor at the mesh origin with zero roll, pitch, and yaw.
    - Set the `Frame Rate` to 60 Hz.
    - Set the `Sensor Name` to "localization" (the default value).

Next we will configure the PID Drive-by-Wire component, which is a component in every AutoSceneGenVehicle. With the vehicle blueprint still open, click on the `Drive By Wire Component` in the component tree. Then under the Details panel, look for the `PID Drive By Wire` tab and make sure the following parameters are set (these should be the default values):
- `Manual Drive`: False (unchecked)
- `Max Manual Drive Speed`: 500
- `Kp Throttle`: 0.0002
- `Kd Throttle`: 0.0007

Finally, make sure this actor is configured to be the default pawn class. Create a new game mode blueprint and set the default pawn class to be the new vehicle blueprint you just created.

#### StructuralSceneActors

The experiments allow for Barberry bushes and Juniper trees to be placed in the scene. Create child blueprints of a [StructuralSceneActor](https://github.com/tsender/AutomaticSceneGeneration/blob/main/Documentation/actors.md) using an appropriate mesh (something similar may suffice, especially if you are just learning how to use this setup). The image below shows what the Barberry bush (left) and Juniper tree (right) look like as used in the experiments.

<img src="documentation/ue4_bush_and_tree.PNG" width="250">

The default values under the Details tab will suffice, but make sure the following values are set:
- `Traversable Height Threshold`: 20
- `Always Traversable`: False (unchecked)


#### AutoSceneGenWorker

The AutoSceneGenWorker actor is the primary actor in UE4 that controls scenario creation and execution. There must exist one of these actors in every level. The following parameters will also need to be set:
- `Worker ID`: 0 (more on this parameter later)
- `Landscape Material`: We used a modified version of the `/Game/StarterContent/Materials/M_Ground_Grass` material in which we set the texture UV scaling to 100.
- `Debug SSASubclasses`: Make sure this TArray is empty, as we do not want additional StructuralSceneActors appearing in the scenarios.
- `Auto Scene Gen Client Name`: "asg_client"

While the landscape parameters are overridden from the RunScenario requests, we will need to set a few parameters appropriately before you press Play so that the vehicle starts on/above the landscape (otherwise it will fall forever):
- `Debug Landscape Subdivisions`: 1
- `Landscape Size`: 6000
- `Landscape Border`: 10000

We also need to make sure the PlayerStart actor is located above the landscape when the game is started. For the PlayerStart actor, set the following Transform parameters in the Details panel to:
- `Location`: x = 0, y = 0, z = 50
- `Rotation`: x = 0, y = 0, z = -45

#### AutoSceneGenLandscape

Each level must have one of these actors. Simply add the actor to the World Outline. The AutoSceneGenWorker will interact with it as required.

#### ROSBridgeParamOverride

Each level should have a ROSBridgeParamOverride actor so it can have its own dedicated rosbridge node.

- `ROSBridge Server Host`: IP address of the computer runnng rosbridge
- `ROSBridge Server Port`: Port number that rosbridge is listening to (we used ports 9100 and above)
- `ROSVersion`: 2
- `Clock Topic Name`: "/clock\<wid\>", where \<wid\> is replaced by the associated Worker ID

### UE4 Setup Part 2: Configuring Parallel Simulations

Since most gaming desktops have a powerful GPU, it is expected that you will be able to run more than one UE editor at once. In our paper, we used 12 independent simulations across two Windows machines, each running six workers. To configure such a setup, first create one level following the above instructions. Save the level as "level_0" (you can use whatever naming conventon you like), as this level will be our first worker. Then, create $n$ copies of this level, "level_1", "level_2", ..., "level_n". Within the $i$th level, set the Worker ID as $i$, set the `ROSBridge Server Port` to $9100 + i$, and set the `Clock Topic Name` (for level 0 this will be "/clock0").

Note: Regardless of how you split up the independent simulations, every active level must have a different Worker ID associated with it (as duplicates will cause problems). Also, all IDs should be consecutive, regardless of the lowest ID you start at. For example, 0,1,2,...,11 or 10,11,12,...,21 are acceptable IDs for using 12 parallel workers.

### ROS Node Setup

#### Autonomy System

The experiments in the paper used multiple variations of a custom autonomy system with a DNN-based perception system classifying image pixels as being part of a traversable or non-traversable object, or to the sky. The dataset is in the folder `astar_trav/Datasets_256x256/NoClouds_Trees_Bushes_SunIncl0-15-30-45-60-75-90`, the network is defined in the `semseg_networks.py`file, the training and prediction code is in the `semantic_segmentation.py` file, and the model used is in the subfolder `models/small_v2`. The dataset contains roughly 300-500 images of scenes with the sun at an inclination angle of 0 to 90 degrees in 15 degree increments. The path planner is an A* voting planner that adds votes to potential obstacle locations (the more votes, the more likely that vertex will be avoided in the planning stage) which affect the edge cost in an A* planner. Then a path follower tracks the latest A* path at a constant speed using a PD controller for steering. The autonomsy system is developed entirely in Python and can be found in the `astar_trav` package.

There are two implementations of the autonomy system. Version 1 consists of two nodes, defined in the files `astar_trav_2D_all_in_on_node.py` (for the A* mapping and planning) and `simple_path_follower_node.py` (for control). In version 2, the A* mapping and planning operations were split into separate nodes, defined in `astar_trav_2D_mapping_node.py` and `astar_trav_2D_planning_node.py`, respectively. Creating this second version also required creating custom messages which are defined in the `astar_trav_msgs` package.

