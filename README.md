# Black-Box Adversarially Compounding Regret Through Evolution (BACRE) Applied to a 2D Off-Road Autonomy System

## Description

This codebase provides an implementation of BACRE as applied to off-road autonomus vehicles (AVs) using the [AutomaticSceneGeneration](https://github.com/tsender/AutomaticSceneGeneration) plugin for UE4. While the implementation of BACRE can be extended to 3D, the AutomaticSceneGeneration currently only provides support for creating 2D off-road scenes (hence the "2D" in the title).

This codebase contains the following ROS packages:
1. `adversarial_scene_gen`: Contains the primary ROS node, an AutoSceneGenClient, for BACRE.
2. `astar_trav`: Contains the code needed to run the A*-traversability autonomy system as used in the experiments in the BACRE paper.
3. `astar_trav_msgs`: Contains custom ROS messages for the `astar_trav` package.
4. `auto_scene_gen_core`: Contains the main ROS nodes and various objects needed to interact with an AutoSceneGenWorker and AutoSceneGenVehicle in UE4.
5. `auto_scene_gen_msgs`: Contains the custom message and service definitions.

Note: For convenience, both `auto_scene_gen_*` packages have been copied directly from the [auto_scene_gen](https://github.com/tsender/auto_scene_gen) ROS2 interface to keep all of the code in a single repository.

## Citation

If you use our work in an academic context, we would greatly appreciate it if you used the following citation:

TODO

## Running the Experiments from the Paper

### Installation

Please refer to the [AutomaticSceneGeneration](https://github.com/tsender/AutomaticSceneGeneration) plugin for UE4 and the corresponding ROS2 interface [auto_scene_gen](https://github.com/tsender/auto_scene_gen) for the software requirements to use this codebase. All of our UE4 simulations were ran on computers running Windows 10 with UE 4.26, and all of the ROS code was developed/executed in docker containers running on linux machines. You can download our docker image with the tag `tsender/tensorflow:gpu-focal-foxy` (you may need to login to your docker account from the command line to pull the image).

### UE4 Setup

In UE4 we need to create an AutoSceneGenVehicle actor, create a few StructuralSceneActors, setup the AutoSceneGenWorker, and then configure rosbridge.

#### AutoSceneGenVehicle

The experiments in the paper use a Polaris MRZR virtual vehicl model (if you do not have access to such a model, then any other car model will suffice for learning how to use the code). Follow these steps to create an [AutoSceneGenVehicle](https://github.com/tsender/AutomaticSceneGeneration/blob/main/Documentation/actors.md) child blueprint, and use the appropriate skeletal mesh for the vehicle model you have at hand.

Now we need to attach the [sensors](https://github.com/tsender/AutomaticSceneGeneration/blob/main/Documentation/sensors.md). The vehicle in the experiments has a forward-facing camera and a localization sensor.
- Open up your vehicle blueprint
- In the blueprint details panel, set the `Vehicle Name` to "vehicle" (the default value).
- Add the camera
    - Add a CompleteCameraSensor at the location (150, 0, 140) centimeters relative to the mesh origin, with a zero roll, pitch, and yaw. Note, these coordinates are in the left-hand coordinate system.
    - Set both the `Image Height` and `Image Width` to 256.
    - Set the `Frame Rate` to 15 Hz.
    - Check the `EnableDepthCam` box to enable the depth camera.
    - Set the `Sensor Name` to "camera" (the default value).
- Add the locaization sensor
    - Add a LocalizationSensor at the mesh origin with zero roll, pitch, and yaw.
    - Set the `Frame Rate` to 60 HZ.
    - Set the `Sensor Name` to "localization" (the default value).

#### StructuralSceneActors

The experiments allow for Barberry bushes and Juniper trees to be placed in the scene. Create child blueprints of a [StructuralSceneActor](https://github.com/tsender/AutomaticSceneGeneration/blob/main/Documentation/actors.md) using an appropriate mesh (something similar may suffice, especially if you are just learning how to use this setup).


#### AutoSceneGenWorker