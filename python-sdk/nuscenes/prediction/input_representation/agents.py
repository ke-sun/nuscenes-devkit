# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
import colorsys
from typing import Any, Dict, List, Tuple, Callable

import cv2
import numpy as np
from pyquaternion import Quaternion

from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import quaternion_yaw
from nuscenes.prediction.input_representation.interface import AgentRepresentation
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_crops, get_rotation_matrix

History = Dict[str, List[Dict[str, Any]]]


def pixels_to_box_corners(row_pixel: int,
                          column_pixel: int,
                          length_in_pixels: float,
                          width_in_pixels: float,
                          yaw_in_radians: float) -> np.ndarray:
    """
    Computes four corners of 2d bounding box for agent.
    The coordinates of the box are in pixels.
    :param row_pixel: Row pixel of the agent.
    :param column_pixel: Column pixel of the agent.
    :param length_in_pixels: Length of the agent.
    :param width_in_pixels: Width of the agent.
    :param yaw_in_radians: Yaw of the agent (global coordinates).
    :return: numpy array representing the four corners of the agent.
    """

    # cv2 has the convention where they flip rows and columns so it matches
    # the convention of x and y on a coordinate plane
    # Also, a positive angle is a clockwise rotation as opposed to counterclockwise
    # so that is why we negate the rotation angle
    coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)

    box = cv2.boxPoints(coord_tuple)

    return box


def get_track_box(annotation: Dict[str, Any],
                  center_coordinates: Tuple[float, float],
                  center_pixels: Tuple[float, float],
                  resolution: float = 0.1) -> np.ndarray:
    """
    Get four corners of bounding box for agent in pixels.
    :param annotation: The annotation record of the agent.
    :param center_coordinates: (x, y) coordinates in global frame
        of the center of the image.
    :param center_pixels: (row_index, column_index) location of the center
        of the image in pixel coordinates.
    :param resolution: Resolution pixels/meter of the image.
    """

    assert resolution > 0

    location = annotation['translation'][:2]
    yaw_in_radians = quaternion_yaw(Quaternion(annotation['rotation']))

    row_pixel, column_pixel = convert_to_pixel_coords(location,
                                                      center_coordinates,
                                                      center_pixels, resolution)

    width = annotation['size'][0] / resolution
    length = annotation['size'][1] / resolution

    # Width and length are switched here so that we can draw them along the x-axis as
    # opposed to the y. This makes rotation easier.
    return pixels_to_box_corners(row_pixel, column_pixel, length, width, yaw_in_radians)


def reverse_history(history: History) -> History:
    """
    Reverse history so that most distant observations are first.
    We do this because we want to draw more recent bounding boxes on top of older ones.
    :param history: result of get_past_for_sample PredictHelper method.
    :return: History with the values reversed.
    """
    return {token: anns[::-1] for token, anns in history.items()}


def add_present_time_to_history(current_time: List[Dict[str, Any]],
                                history: History) -> History:
    """
    Adds the sample annotation records from the current time to the
    history object.
    :param current_time: List of sample annotation records from the
        current time. Result of get_annotations_for_sample method of
        PredictHelper.
    :param history: Result of get_past_for_sample method of PredictHelper.
    :return: History with values from current_time appended.
    """

    for annotation in current_time:
        token = annotation['instance_token']

        if token in history:

            # We append because we've reversed the history
            history[token].append(annotation)

        else:
            history[token] = [annotation]

    return history


def fade_color(color: Tuple[int, int, int],
               step: int,
               total_number_of_steps: int) -> Tuple[int, int, int]:
    """
    Fades a color so that past observations are darker in the image.
    :param color: Tuple of ints describing an RGB color.
    :param step: The current time step.
    :param total_number_of_steps: The total number of time steps
        the agent has in the image.
    :return: Tuple representing faded rgb color.
    """

    LOWEST_VALUE = 0.4

    if step == total_number_of_steps:
        return color

    hsv_color = colorsys.rgb_to_hsv(*color)

    increment = (float(hsv_color[2])/255. - LOWEST_VALUE) / total_number_of_steps

    new_value = LOWEST_VALUE + step * increment

    new_rgb = colorsys.hsv_to_rgb(float(hsv_color[0]),
                                  float(hsv_color[1]),
                                  new_value * 255.)
    return new_rgb


def default_colors(category_name: str) -> Tuple[int, int, int]:
    """
    Maps a category name to an rgb color (without fading).
    :param category_name: Name of object category for the annotation.
    :return: Tuple representing rgb color.
    """

    if 'vehicle' in category_name:
        return 255, 255, 0  # yellow
    elif 'object' in category_name:
        return 204, 0, 204  # violet
    elif 'human' in category_name or 'animal' in category_name:
        return 255, 153, 51  # orange
    elif category_name == 'ego':
        return 0, 255, 255 # cyan
    else:
        raise ValueError(f"Cannot map {category_name} to a color.")


def draw_agent_boxes(center_agent_annotation: Dict[str, Any],
                     center_agent_pixels: Tuple[float, float],
                     agent_history: History,
                     base_image: np.ndarray,
                     get_color: Callable[[str], Tuple[int, int, int]],
                     resolution: float = 0.1) -> None:
    """
    Draws past sequence of agent boxes on the image.
    :param center_agent_annotation: Annotation record for the agent
        that is in the center of the image.
    :param center_agent_pixels: Pixel location of the agent in the
        center of the image.
    :param agent_history: History for all agents in the scene.
    :param base_image: Image to draw the agents in.
    :param get_color: Mapping from category_name to RGB tuple.
    :param resolution: Size of the image in pixels / meter.
    :return: None.
    """

    agent_x, agent_y = center_agent_annotation['translation'][:2]

    for instance_token, annotations in agent_history.items():

        num_points = len(annotations)

        for i, annotation in enumerate(annotations):

            box = get_track_box(annotation, (agent_x, agent_y), center_agent_pixels, resolution)

            if instance_token == center_agent_annotation['instance_token']:
                color = (255, 0, 0)
            else:
                color = get_color(annotation['category_name'])

            # Don't fade the colors if there is no history
            if num_points > 1:
                color = fade_color(color, i, num_points - 1)

            cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

def get_present_for_ego(helper: PredictHelper, sample_token: str) -> Dict[str, Any]:
    """
    Get the present data for the ego vehicle.

    :param: helper The PredictHelper object.
    :param: sample_token The token of the present sample.
    :return: The present ego pose data.
    """

    nusc = helper.data

    sample = nusc.get('sample', sample_token)
    cam_front_data_token = sample['data']['CAM_FRONT']
    cam_front_data = nusc.get('sample_data', cam_front_data_token)
    ego_pose_token = cam_front_data['ego_pose_token']

    return nusc.get('ego_pose', ego_pose_token)

def get_past_for_ego(helper: PredictHelper, sample_token: str, seconds: int) -> List[Dict[str, Any]]:
    """
    Get the past data for the ego vehicle.

    :param: helper The PredictHelper object.
    :param: sample_token The token of the present sample.
    :param: seconds The duration to look back in the past.
    :return: A list of past ego pose data.
             The present ego pose data (associated with the sample_token) is not included in the list.
             The most distant ego pose in time is at index 0 of the returned list.
    """

    nusc = helper.data

    sample = nusc.get('sample', sample_token)
    present_time = sample['timestamp']

    past_ego_pose = []

    while True:

        sample_token = sample['prev']

        # Break if there is no more past data.
        if not sample_token: break

        sample = nusc.get('sample', sample_token)

        # Break if enough history data has been retrieved.
        if helper._absolute_time_diff(sample['timestamp'], present_time) > seconds: break

        # Get the ego pose data associated with this sample.
        cam_front_data_token = sample['data']['CAM_FRONT']
        cam_front_data = nusc.get('sample_data', cam_front_data_token)
        ego_pose_token = cam_front_data['ego_pose_token']
        past_ego_pose.append(nusc.get('ego_pose', ego_pose_token))

    # Reverse the data so that the most distant ego pose in time is at index 0.
    past_ego_pose.reverse()

    return past_ego_pose

def get_ego_track_box(ego_pose: Dict[str, Any],
                      center_coordinates: Tuple[float, float],
                      center_pixels: Tuple[float, float],
                      resolution: float = 0.1) -> np.ndarray:
    """
    Get four corners of bounding box for agent in pixels.
    :param ego_pose: The ego pose data.
    :param center_coordinates: (x, y) coordinates in global frame
        of the center of the image.
    :param center_pixels: (row_index, column_index) location of the center
        of the image in pixel coordinates.
    :param resolution: Resolution pixels/meter of the image.
    """

    assert resolution > 0

    location = ego_pose['translation'][:2]
    yaw_in_radians = quaternion_yaw(Quaternion(ego_pose['rotation']))

    row_pixel, column_pixel = convert_to_pixel_coords(location,
                                                      center_coordinates,
                                                      center_pixels, resolution)

    width = 1.8 / resolution
    length = 4.0 / resolution

    # Width and length are switched here so that we can draw them along the x-axis as
    # opposed to the y. This makes rotation easier.
    return pixels_to_box_corners(row_pixel, column_pixel, length, width, yaw_in_radians)

def draw_ego_boxes(center_agent_annotation: Dict[str, Any],
                   center_agent_pixels: Tuple[float, float],
                   ego_history: List[Dict[str, Any]],
                   base_image: np.ndarray,
                   get_color: Callable[[str], Tuple[int, int, int]],
                   resolution: float = 0.1) -> None:
    """
    Draws past sequence of agent boxes on the image.
    :param center_agent_annotation: Annotation record for the agent
        that is in the center of the image.
    :param center_agent_pixels: Pixel location of the agent in the
        center of the image.
    :param ego_history: History for the ego in the scene.
    :param base_image: Image to draw the agents in.
    :param get_color: Mapping from category_name to RGB tuple.
    :param resolution: Size of the image in pixels / meter.
    :return: None.
    """

    agent_x, agent_y = center_agent_annotation['translation'][:2]

    num_points = len(ego_history)

    for i, ego_pose in enumerate(ego_history):

        box = get_ego_track_box(ego_pose, (agent_x, agent_y), center_agent_pixels, resolution)

        color = get_color('ego')

        # Don't fade the colors if there is no history
        if num_points > 1:
            color = fade_color(color, i, num_points - 1)

        cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

    return


class AgentBoxesWithFadedHistory(AgentRepresentation):
    """
    Represents the past sequence of agent states as a three-channel
    image with faded 2d boxes.
    """

    def __init__(self, helper: PredictHelper,
                 seconds_of_history: float = 2,
                 frequency_in_hz: float = 2,
                 resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25,
                 color_mapping: Callable[[str], Tuple[int, int, int]] = None,
                 include_ego: bool = True):

        self.helper = helper
        self.seconds_of_history = seconds_of_history
        self.frequency_in_hz = frequency_in_hz

        if not resolution > 0:
            raise ValueError(f"Resolution must be positive. Received {resolution}.")

        self.resolution = resolution

        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

        if not color_mapping:
            color_mapping = default_colors

        self.color_mapping = color_mapping

        self.include_ego = include_ego

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        Draws agent boxes with faded history into a black background.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :return: np.ndarray representing a 3 channel image.
        """

        # Taking radius around track before to ensure all actors are in image
        buffer = max([self.meters_ahead, self.meters_behind,
                      self.meters_left, self.meters_right]) * 2

        image_side_length = int(buffer/self.resolution)

        # We will center the track in the image
        central_track_pixels = (image_side_length / 2, image_side_length / 2)

        base_image = np.zeros((image_side_length, image_side_length, 3))

        history = self.helper.get_past_for_sample(sample_token,
                                                  self.seconds_of_history,
                                                  in_agent_frame=False,
                                                  just_xy=False)
        history = reverse_history(history)

        present_time = self.helper.get_annotations_for_sample(sample_token)

        history = add_present_time_to_history(present_time, history)

        center_agent_annotation = self.helper.get_sample_annotation(instance_token, sample_token)

        draw_agent_boxes(center_agent_annotation, central_track_pixels,
                         history, base_image, resolution=self.resolution, get_color=self.color_mapping)

        # Draw past of the ego vehicle.
        if self.include_ego:

            ego_history = get_past_for_ego(self.helper, sample_token, self.seconds_of_history)

            ego_present_time = get_present_for_ego(self.helper, sample_token)

            ego_history.append(ego_present_time)

            draw_ego_boxes(center_agent_annotation, central_track_pixels,
                           ego_history, base_image, resolution=self.resolution, get_color=self.color_mapping)

        center_agent_yaw = quaternion_yaw(Quaternion(center_agent_annotation['rotation']))
        rotation_mat = get_rotation_matrix(base_image.shape, center_agent_yaw)

        rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1],
                                                                  base_image.shape[0]))

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind,
                                       self.meters_left, self.meters_right, self.resolution,
                                       image_side_length)

        return rotated_image[row_crop, col_crop].astype('uint8')
