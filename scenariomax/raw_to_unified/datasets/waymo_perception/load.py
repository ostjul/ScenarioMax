import os
import numpy as np

import scenariomax.raw_to_unified.datasets.waymo.waymo_protos.scenario_pb2 as scenario_pb2
import scenariomax.raw_to_unified.datasets.waymo.waymo_protos.dataset_pb2 as dataset_pb2
from scenariomax.raw_to_unified.datasets.waymo_perception.utils import Track, Scenario, State
from scenariomax import logger_utils
from scenariomax.raw_to_unified.datasets.waymo import utils as waymo_utils
from scenariomax.tf_utils import get_tensorflow


logger = logger_utils.get_logger(__name__)


def get_waymo_perception_scenarios(data_path, start_index: int = 0, num: int | None = None):
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    logger.debug("Reading raw data")
    file_list = os.listdir(data_path)

    if num is None:
        num = len(file_list) - start_index

    assert len(file_list) >= start_index + num and start_index >= 0, (
        f"No sufficient files ({len(file_list)}) in raw_data_directory. need: {num}, start: {start_index}"
    )

    file_list = file_list[start_index : start_index + num]
    num_files = len(file_list)
    all_result = [os.path.join(data_path, f) for f in file_list]
    logger.debug(f"Find {num_files} waymo files")

    return all_result


def preprocess_waymo_perception_scenarios(files, worker_index):
    """Convert the waymo perception files into scenario_pb2. This happens in each worker.

    Args:
        files: list of files to be converted
        worker_index: index of the worker

    Returns:
        Generator of scenario_pb2.Scenario
    """
    tf = get_tensorflow()

    for file in files:
        file_path = os.path.join(file)
        if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
            continue
        frames = []
        count = 0
        for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
            # Load only 2 frames. Note that using too many frames may be slow to display.
            frame = dataset_pb2.Frame.FromString(data)
            frames.append(frame)
            count += 1
            # if count == 20: 
            #     break
            # scenario.ParseFromString(data)

        timestamps_seconds = [(frame.timestamp_micros - frames[0].timestamp_micros) *  1e-6 for frame in frames]

        tracks, tracks_to_predict, sdc_track_index = load_tracks(frames)

        dynamic_map_states = [[] for frame in frames]

        scenario = Scenario({
            # The unique ID for this scenario.
            "scenario_id": frame.context.name[:19],
            # Timestamps corresponding to the track states for each step in the scenario.
            # The length of this field is equal to tracks[i].states_size() for all tracks
            # i and equal to the length of the dynamic_map_states_field.
            "timestamps_seconds": timestamps_seconds,
            # The index into timestamps_seconds for the current time. All time stepsafter this index are
            # future data to be predicted. All steps before this index are history data.
            "current_time_index": 40,
            # The index into the tracks field of the autonomous vehicle object.
            "sdc_track_index": sdc_track_index,
            # A list of objects IDs in the scene detected to have interactive behavior.
            # The objects in this list form an interactive group. These IDs correspond
            # to IDs in the tracks field above.
            "objects_of_interest": [],
            # A list of tracks to generate predictions for. For the challenges, exactly
            # these objects must be predicted in each scenario for test and validation
            # submissions. This field is populated in the training set only as a
            # suggestion of objects to train on.
            "tracks_to_predict": tracks_to_predict,
            # Tracks for all objects in the scenario. All object tracks in all scenarios
            # in the dataset have the same number of object states. In this way, the
            # tracks field forms a 2 dimensional grid with objects on one axis and time
            # on the other. Each state can be associated with a timestamp in the
            # 'timestamps_seconds' field by its index. E.g., tracks[i].states[j] indexes
            # the i^th agent's state at time timestamps_seconds[j].
            "tracks": tracks,
            # The set of static map features for the scenario.
            "map_features": frames[0].map_features,
            # The dynamic map states in the scenario (e.g. traffic signal states).
            # This field has the same length as timestamps_seconds. Each entry in this
            # field can be associated with a timestamp in the 'timestamps_seconds' field
            # by its index. E.g., dynamic_map_states[i] indexes the dynamic map state at
            # time timestamps_seconds[i].
            "dynamic_map_states": dynamic_map_states,
            "compressed_frame_laser_data": [],
            "frame_camera_tokens": []
        })


        scenario.scenario_id = scenario.scenario_id + waymo_utils.SPLIT_KEY + file

        yield scenario


def load_tracks(frames):
    """Load tracks from laser labels.
    Args:
        frames: list of Frame objects containing laser_labels
    Returns:
        tracks: list of tracks, each track follows this structure:
                {
                    "id": int,
                    "object_type": str,
                    "states": 
                        [
                            {
                            "center_x": float,
                            "center_y": float,
                            "center_z": float,
                            "heading": float,
                            "velocity_x": float,
                            "velocity_y": float,
                            "valid": bool,
                            "length": float,
                            "width": float,
                            "height": float,
                            }
                        ]
                }
        tracks_to_predict: list of track ids to predict
    """
    from scenariomax.raw_to_unified.datasets.waymo import types as waymo_types
    
    # Dictionary to collect all observations by track ID
    track_observations = {}
    num_timesteps = len(frames)
    
    # First pass: collect all observations by track ID and timestep
    for timestep, frame in enumerate(frames):
        # Extract ego pose and add as a track
        ego_pose = np.reshape(np.array(frame.pose.transform, np.float32), [4, 4])
        
        # Extract ego center (translation components)
        ego_center_x = ego_pose[0, 3]
        ego_center_y = ego_pose[1, 3]
        ego_center_z = ego_pose[2, 3]
        
        # Extract ego heading from rotation matrix
        # Heading is the yaw angle (rotation around z-axis)
        ego_heading = np.arctan2(ego_pose[1, 0], ego_pose[0, 0])

        im_close_idx = np.argmin([(frame.timestamp_micros * 1e-6 - im.pose_timestamp) for im in frame.images])
        im_close = frame.images[im_close_idx]

        ego_velocity = np.array([im_close.velocity.v_x, im_close.velocity.v_y, im_close.velocity.v_z], np.float32)

        ego_length = 4.7
        ego_width = 2.3
        ego_height = 1.6

        ego_track_id = 0  # Using 0 as the ID for the ego vehicle
        ego_type = waymo_types.get_agent_type(1)  # Type 1 is VEHICLE

        # Initialize ego track if not seen before
        if ego_track_id not in track_observations:
            track_observations[ego_track_id] = {
                "id": ego_track_id,
                "object_type": ego_type,
                "observations": {}  # timestep -> observation dict
            }
        
        # Add ego observation for this timestep
        ego_observation = {
            "center_x": float(ego_center_x),
            "center_y": float(ego_center_y),
            "center_z": float(ego_center_z),
            "heading": float(ego_heading),
            "velocity_x": float(ego_velocity[0]),
            "velocity_y": float(ego_velocity[1]),
            "valid": True,  # Ego is always valid
            "length": ego_length,
            "width": ego_width,
            "height": ego_height,
        }
        
        track_observations[ego_track_id]["observations"][timestep] = ego_observation

        for label in frame.laser_labels:
            track_id = label.id
            
            # Initialize track if not seen before
            if track_id not in track_observations:
                track_observations[track_id] = {
                    "id": track_id,
                    "object_type": waymo_types.get_agent_type(label.type),
                    "observations": {}  # timestep -> observation dict
                }
            
            # Extract observation data from label (in ego coordinates)
            label_center_x = label.box.center_x if label.box else 0.0
            label_center_y = label.box.center_y if label.box else 0.0
            label_center_z = label.box.center_z if label.box else 0.0
            label_heading = label.box.heading if label.box else 0.0
            label_velocity_x = label.metadata.speed_x if label.metadata else 0.0
            label_velocity_y = label.metadata.speed_y if label.metadata else 0.0
            
            # Transform position from ego coordinates to world coordinates
            label_position = np.array([label_center_x, label_center_y, label_center_z, 1.0])
            world_position = ego_pose @ label_position
            
            # Transform heading from ego coordinates to world coordinates
            # The heading in ego frame needs to be rotated by the ego's heading (already computed)
            world_heading = label_heading + ego_heading  # ego_heading is the ego's world heading
            
            # Transform velocity from ego coordinates to world coordinates
            # Rotate velocity vector by ego's rotation matrix
            label_velocity_3d = np.array([label_velocity_x, label_velocity_y, 0.0])
            ego_rotation = ego_pose[:3, :3]
            world_velocity_3d = ego_rotation @ label_velocity_3d
            
            observation = {
                "center_x": float(world_position[0]),
                "center_y": float(world_position[1]),
                "center_z": float(world_position[2]),
                "heading": float(world_heading),
                "velocity_x": float(world_velocity_3d[0]),
                "velocity_y": float(world_velocity_3d[1]),
                "valid": True,  # Label exists, so it's valid
                "length": label.box.length if label.box else 1.0,
                "width": label.box.width if label.box else 1.0,
                "height": label.box.height if label.box else 1.0,
            }
            
            track_observations[track_id]["observations"][timestep] = observation
    
    # Second pass: create complete tracks with all timesteps
    tracks = []
    
    for track_id, track_data in track_observations.items():
        # Create states list for all timesteps
        states = []
        
        for timestep in range(num_timesteps):
            if timestep in track_data["observations"]:
                # Use actual observation
                states.append(State(track_data["observations"][timestep]))
            else:
                # Create non-valid default state
                default_state = {
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "center_z": 0.0,
                    "heading": 0.0,
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                    "valid": False,  # Mark as invalid
                    "length": 1.0,
                    "width": 1.0,
                    "height": 1.0,
                }
                states.append(State(default_state))

        # Create final track structure
        track = {
            "id": track_data["id"],
            "object_type": track_data["object_type"],
            "states": states
        }
        
        tracks.append(track)
        
        # For now, we don't have specific tracks_to_predict information
        # from the perception dataset, so we'll leave this empty
        # This would need to be populated based on specific requirements
    
    return_tracks = []
    tracks_to_predict = []
    for idx, tr in enumerate(tracks):
        tr = Track({
            "id": idx,
            "object_type": tr["object_type"],
            "states": tr["states"]
        })
        return_tracks.append(tr)

        # if idx > 0 and tr.object_type in [ "VEHICLE", "PEDESTRIAN", "CYCLIST"]:
        #     tracks_to_predict.append(idx)

    ego_index = 0
    return return_tracks, tracks_to_predict, ego_index


# def count_waymo_perception_scenarios(files):
#     """Count the number of waymo perception scenarios in the files.

#     Args:
#         files: list of files to be counted

#     Returns:
#         int: number of waymo scenarios
#     """
#     tf = get_tensorflow()
#     count = 0
#     for file in files:
#         file_path = os.path.join(file)
#         if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
#             continue
#         count += sum(1 for _ in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator())

#     return count
