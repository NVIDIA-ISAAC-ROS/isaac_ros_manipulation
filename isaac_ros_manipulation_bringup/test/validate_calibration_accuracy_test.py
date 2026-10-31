# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Validate the calibration accuracy by performing various tests."""

from abc import ABC, abstractmethod
import json
import os
from typing import Optional, TypedDict
import unittest

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from scipy.spatial.transform import Rotation as scipy_rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage


DEFAULT_NUM_STILL_FRAMES = 10
DEFAULT_STILL_PIXEL_THRESHOLD = 5
SKIP_TEST = False
if os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() != 'manual_on_robot':
    print('Skipping test: ENABLE_MANIPULATOR_TESTING is not set to '
          'manual_on_robot')
    print('To run this test, set: export ENABLE_MANIPULATOR_TESTING=manual_on_robot')
    SKIP_TEST = True

JSON_FILE_PATH = os.environ.get('CALIBRATION_TRANSFORMS_FILE', None)
if JSON_FILE_PATH is None:
    EXTRINSICS_CALIBRATIONS = None
else:
    EXTRINSICS_CALIBRATIONS = json.load(open(JSON_FILE_PATH))


NUM_SQUARE_HEIGHT = os.environ.get('NUM_SQUARE_HEIGHT', None)
NUM_SQUARE_WIDTH = os.environ.get('NUM_SQUARE_WIDTH', None)
LONGER_SIDE_M = os.environ.get('LONGER_SIDE_M', None)
MARKER_SIZE_M = os.environ.get('MARKER_SIZE_M', None)

if (
    NUM_SQUARE_HEIGHT is None
    or NUM_SQUARE_WIDTH is None
    or LONGER_SIDE_M is None
    or MARKER_SIZE_M is None
):
    if not SKIP_TEST:
        raise ValueError('NUM_SQUARE_HEIGHT, NUM_SQUARE_WIDTH, LONGER_SIDE_M, '
                         'and MARKER_SIZE_M must be set')


class Translation(TypedDict):
    x: float
    y: float
    z: float


class Quaternion(TypedDict):
    x: float
    y: float
    z: float
    w: float


class Pose(TypedDict):
    position: Translation
    orientation: Quaternion
    identifier: str
    frame_id: str
    pose_name_for_tf: str


class Intrinsics():

    def __init__(self, camera_matrix, optical_T_rectified, image_width, image_height):
        self.camera_matrix = camera_matrix
        self.optical_T_rectified = optical_T_rectified
        self.width = image_width
        self.height = image_height


class TargetDetector(ABC):
    """
    Target detector base class.

    Yaml input expected to have the following format:

    target_type: 'checkerboard'     # [checkerboard]
    num_squares_height: 9           # Number of squares in Y dimension
    num_squares_width: 12           # Number of squares in X dimension
    longer_side_m: 0.36             # Length in meters of the longest side of the checker pattern
    num_still_frames: 10            # Number of consecutive still frames to use for pose estimation
    still_image_pixel_threshold: 5  # Maximum pixel detection difference to consider still frame
    """

    def __init__(self, target_description):
        if SKIP_TEST:
            return
        self.num_still_frames = target_description.get('num_still_frames',
                                                       DEFAULT_NUM_STILL_FRAMES)
        self.still_image_pixel_threshold = target_description.get('still_image_pixel_threshold',
                                                                  DEFAULT_STILL_PIXEL_THRESHOLD)
        if self.still_image_pixel_threshold < 0:
            raise ValueError('Error parsing target description: still_image_pixel_threshold must '
                             'be positive')
        if self.num_still_frames < 0:
            raise ValueError('Error parsing target description: num_still_frames must be positive')
        if self.num_still_frames == 0:
            print(f'num_still_frames == {self.num_still_frames}. Not filtering '
                  'static frames: Using all matching frames')

    @abstractmethod
    def detect_target(self, image_bgr, refine_subpixel):
        """Detect target in image and its 2D corners."""
        pass

    @abstractmethod
    def estimate_pose(self, corners_2d, intrinsics, refine_pnp, pnp_flags):
        """Estimate camera_optical_T_target, given 2D corners in an image."""
        pass

    @abstractmethod
    def estimate_corners_3d(self, corners_2d, intrinsics, refine_pnp, pnp_flags):
        """Compute 3D points of the target in camera frame, given the rectified 2D detections."""
        pass

    @abstractmethod
    def plot_corners(self, image_bgr, corners_2d, detection_text, text_origen):
        """Plot corners and axes of transformation, or the text 'Target not detected!'."""
        pass

    @abstractmethod
    def get_keypoint_count(self):
        """Return the number of corners in the calibration target."""
        pass

    def get_num_still_frames(self):
        return self.num_still_frames

    def get_still_image_pixel_threshold(self):
        return self.still_image_pixel_threshold


class CharucoBoardDetector(TargetDetector):

    def __init__(self, target_description):
        super().__init__(target_description)
        num_squares_height = target_description['num_squares_height']
        num_squares_width = target_description['num_squares_width']
        longer_side_m = target_description['longer_side_m']
        self.checkerboard_dimensions = (num_squares_width - 1, num_squares_height - 1)
        self.corners_3d = self._compute_checkerboard_3d_points(
            num_squares_height, num_squares_width, longer_side_m,
            target_description['marker_size_m'])

    def _compute_checkerboard_3d_points(self, num_squares_height, num_squares_width,
                                        longer_side_m, marker_size_m) -> np.ndarray:
        """Compute points in the format [[0,0,0],[1,0,0],[2,0,0],...,[M,N,0]] * square_size."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        # longer_side_m spans the longest side of the pattern, e.g. 0.28 m / 14 squares = 20 mm.
        checkerboard_square_size_m = longer_side_m / max(num_squares_width, num_squares_height)
        all_obj_pts = cv2.aruco.CharucoBoard_create(
            squaresX=num_squares_width, squaresY=num_squares_height,
            squareLength=checkerboard_square_size_m,
            markerLength=marker_size_m, dictionary=aruco_dict)
        self.board = all_obj_pts
        return all_obj_pts.chessboardCorners

    def get_keypoint_count(self):
        raise NotImplementedError('Charuco board detector not implemented')

    def detect_target(self, image_bgr, refine_subpixel=False) -> list[np.ndarray]:
        """
        Detect target in image and its 2D corners.

        Returns
        -------
            list of 2D corners, each element is of shape (1,4,2)

        """
        if len(image_bgr.shape) < 3:
            image_gray = image_bgr
        else:
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_50 | cv2.aruco.DICT_5X5_100
            | cv2.aruco.DICT_5X5_250 | cv2.aruco.DICT_5X5_1000)
        detectorParams = cv2.aruco.DetectorParameters_create()
        corners_2d, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
            image_gray, dictionary, parameters=detectorParams)
        if not corners_2d:
            print('Target not detected')
            return None
        if refine_subpixel:
            detect_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            detect_search_window = (5, 5)
            disable_zero_zone = (-1, -1)
            for i in range(len(corners_2d)):
                # Updates in place.
                cv2.cornerSubPix(image_gray, corners_2d[i], winSize=detect_search_window,
                                 zeroZone=disable_zero_zone, criteria=detect_criteria)

        return corners_2d, marker_ids

    def estimate_pose_new_all_corners(self, charuco_corners, charuco_ids,
                                      cameraMatrix, distCoeffs):
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board, cameraMatrix, None, None, None
        )
        camera_optical_T_target = np.eye(4)
        camera_optical_T_target[:3, :3] = scipy_rotation.from_rotvec(
            rvec.reshape((-1, ))).as_matrix()
        camera_optical_T_target[:3, 3] = tvec.reshape((-1, ))
        return valid, rvec, tvec, camera_optical_T_target, self.board.chessboardCorners

    def estimate_pose(self, corners_2d, intrinsics, refine_pnp='VVS',
                      pnp_flags=cv2.SOLVEPNP_ITERATIVE):
        ret, rvec, tvec = cv2.solvePnP(self.corners_3d, corners_2d, intrinsics.camera_matrix, None,
                                       flags=pnp_flags)

        def to_4x4_matrix(rotation_matrix, translation=np.array([0, 0, 0])):
            """Convert 3d rotation matrix and translation vector to homogeneous matrix in SE3."""
            homogeneous_transform = np.eye(4, dtype=rotation_matrix.dtype)
            homogeneous_transform[:3, :3] = rotation_matrix
            homogeneous_transform[:3, 3] = translation.reshape((-1, ))
            return homogeneous_transform

        def rodriquez_to_4x4_matrix(rotation_rodrigez, translation=np.array([0, 0, 0])):
            """Convert a rodriguez vector in radians and translation to homogeneous matrix SE3."""
            rotation = scipy_rotation.from_rotvec(rotation_rodrigez.reshape((-1, ))).as_matrix()
            return to_4x4_matrix(rotation, translation)

        if not ret:
            self.logger.warning('PnP failed')
            return None
        if refine_pnp:
            stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            if refine_pnp == 'LM':
                rvec, tvec = cv2.solvePnPRefineLM(self.corners_3d, corners_2d,
                                                  intrinsics.camera_matrix, None, rvec, tvec,
                                                  stop_criteria)
            elif refine_pnp == 'VVS':
                rvec, tvec = cv2.solvePnPRefineVVS(self.corners_3d, corners_2d,
                                                   intrinsics.camera_matrix, None, rvec, tvec,
                                                   stop_criteria)
            else:
                raise ValueError('Unexpected PNP refine type. Supported: LM, VVS')
        camera_rect_T_target = rodriquez_to_4x4_matrix(rvec, tvec)
        camera_optical_T_target = intrinsics.optical_T_rectified @ camera_rect_T_target
        return camera_optical_T_target

    def estimate_corners_3d(self, corners_2d, intrinsics, refine_pnp='VVS',
                            pnp_flags=cv2.SOLVEPNP_ITERATIVE):
        camera_optical_T_target = self.estimate_pose(corners_2d, intrinsics, refine_pnp, pnp_flags)
        return self.corners_3d @ camera_optical_T_target[:3, :3].T + camera_optical_T_target[:3, 3]

    def get_corners_and_ids(self, corners_2d, marker_ids, image_bgr):
        image_with_corners = image_bgr.copy()
        rtval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners_2d, marker_ids, image_with_corners, board=self.board
        )
        if not rtval:
            raise ValueError('Failed to interpolate corners')
        return charuco_corners, charuco_ids

    def plot_corners(self, image_bgr, corners_2d, marker_ids, detection_text='',
                     text_origen=30) -> np.ndarray:
        """Plot corners and axes of transformation, or the text 'Target not detected!'."""
        plotted_image = image_bgr.copy()
        image_with_corners = image_bgr.copy()
        if corners_2d is not None:
            cv2.aruco.drawDetectedMarkers(plotted_image, corners_2d, marker_ids)

            cv2.putText(plotted_image, detection_text,
                        org=(text_origen, plotted_image.shape[0] - text_origen),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                        thickness=2)

            rtval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners_2d, marker_ids, image_with_corners, board=self.board
            )
            cv2.aruco.drawDetectedCornersCharuco(charucoCorners=charuco_corners,
                                                 image=image_with_corners)
            # Now draw detected Conerns
            # cv2.imwrite(f'{self._output_dir}/image_with_corners.png', image_with_corners)
        else:
            cv2.putText(plotted_image, 'Target not detected!',
                        org=(text_origen, plotted_image.shape[0] - text_origen),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0),
                        thickness=2)
        return plotted_image, image_with_corners


class ValidateCalibrationAccuracyTest(unittest.TestCase):
    """
    Unit tests for validating the calibration accuracy.

    This test will take a rosbag, and process it to get the rgb image and camera info.
    It will also take in rgb image of infra1 and camera info of infra1.


    It will also look at tf and tf static transforms and store those transform trees.
    """

    rosbag_path: str
    rgb_image: np.ndarray
    infra1_rgb_image: np.ndarray
    infra1_camera_info: CameraInfo
    rgb_camera_info: CameraInfo
    infra2_rgb_image: np.ndarray
    infra2_camera_info: CameraInfo
    _output_dir: str
    tf_transforms: dict[str, TransformStamped]
    static_tf_transforms: dict[str, TransformStamped]

    @classmethod
    def _get_images_and_camera_info(
            cls
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[CameraInfo], Optional[CameraInfo]]:
        """
        Read the first RGB image and CameraInfo from the rosbag.

        If infra1 topics are absent, return None for infra1 image and infra1 camera info.

        Returns
        -------
            (rgb_image_np, infra1_image_np, infra1_camera_info_msg, rgb_camera_info_msg)

        """
        bag_uri = cls.rosbag_folder_path  # folder containing the mcap/metadata.yaml
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=bag_uri, storage_id='mcap'),
            ConverterOptions(input_serialization_format='cdr',
                             output_serialization_format='cdr'),
        )

        bridge = CvBridge()

        rgb_np: Optional[np.ndarray] = None
        rgb_info: Optional[CameraInfo] = None
        infra1_np: Optional[np.ndarray] = None
        infra1_info: Optional[CameraInfo] = None
        infra2_np: Optional[np.ndarray] = None
        infra2_info: Optional[CameraInfo] = None

        # Known topics in the provided bag metadata
        RGB_IMAGE_TOPIC = '/camera_1/color/image_raw'
        RGB_INFO_TOPIC = '/camera_1/color/camera_info'
        TF_TOPIC = '/tf'
        STATIC_TF_TOPIC = '/tf_static'

        # Common infra1 candidates (bag may not have these; we’ll return None if missing)
        INFRA1_IMAGE_TOPICS = ['/camera_1/infra1/image_rect_raw_drop']
        INFRA2_IMAGE_TOPICS = ['/camera_1/infra2/image_rect_raw_drop']
        INFRA1_INFO_TOPICS = ['/camera_1/infra1/camera_info_drop']
        INFRA2_INFO_TOPICS = ['/camera_1/infra2/camera_info_drop']

        idx = 0

        all_topics = set()

        while reader.has_next():
            topic, data, _ = reader.read_next()
            all_topics.add(topic)

            if topic == RGB_IMAGE_TOPIC:
                img_msg = deserialize_message(data, Image)
                rgb_np = np.array(bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8'))

            elif topic == RGB_INFO_TOPIC and rgb_info is None:
                rgb_info = deserialize_message(data, CameraInfo)

            elif topic in INFRA1_IMAGE_TOPICS:
                img_msg = deserialize_message(data, Image)
                # Infra is usually mono8/y16; accept as-is and convert to numpy
                # Without forcing encoding to preserve the native format.
                infra1_np = np.array(bridge.imgmsg_to_cv2(img_msg))
            elif topic in INFRA2_IMAGE_TOPICS:
                img_msg = deserialize_message(data, Image)
                infra2_np = np.array(bridge.imgmsg_to_cv2(img_msg))
                idx += 1

            elif topic in INFRA1_INFO_TOPICS and infra1_info is None:
                infra1_info = deserialize_message(data, CameraInfo)

            elif topic in INFRA2_INFO_TOPICS and infra2_info is None:
                infra2_info = deserialize_message(data, CameraInfo)

            elif topic == TF_TOPIC:
                tf_msg = deserialize_message(data, TFMessage)
                for transform in tf_msg.transforms:
                    str_key = f'{transform.header.frame_id}<-{transform.child_frame_id}'
                    if str_key in cls.tf_transforms:
                        cls.tf_transforms[str_key].append(transform)
                    else:
                        cls.tf_transforms[str_key] = [transform]

            elif topic == STATIC_TF_TOPIC:
                tf_msg = deserialize_message(data, TFMessage)
                for transform in tf_msg.transforms:
                    str_key = f'{transform.header.frame_id}<-{transform.child_frame_id}'
                    if str_key in cls.static_tf_transforms:
                        cls.static_tf_transforms[str_key].append(transform)
                    else:
                        cls.static_tf_transforms[str_key] = [transform]
        return rgb_np, infra1_np, infra1_info, rgb_info, infra2_np, infra2_info

    @classmethod
    def setUpClass(cls) -> None:
        """Run once before all tests."""
        if SKIP_TEST:
            return
        cls.json_folder_path = os.environ.get('CALIBRATION_VALIDATION_OUTPUT_DIR')
        if cls.json_folder_path is None:
            raise ValueError('CALIBRATION_VALIDATION_OUTPUT_DIR is not set')
        cls._output_dir = f'{cls.json_folder_path}/calibration_validation_results'
        os.makedirs(cls._output_dir, exist_ok=True)
        cls.rosbag_folder_path = f'{cls.json_folder_path}/rosbag'
        cls.rosbag_folders = [
            f for f in os.listdir(cls.rosbag_folder_path) if os.path.isdir(
                os.path.join(cls.rosbag_folder_path, f))
        ]
        if len(cls.rosbag_folders) == 0:
            raise ValueError('No rosbag folders found in the rosbag folder')

        # Bag folders are timestamped (YYYY-mm-dd_HH-MM-SS), so the last sorted one is newest.
        cls.rosbag_folder_path = f'{cls.rosbag_folder_path}/{sorted(cls.rosbag_folders)[-1]}'

        cls.tf_transforms = {}
        cls.static_tf_transforms = {}

        objs = cls._get_images_and_camera_info()
        (cls.rgb_image, cls.infra1_image, cls.infra1_camera_info,
         cls.rgb_camera_info, cls.infra2_image, cls.infra2_camera_info) = objs

        cls.save_camera_intrinsic_matrices()

        # Save the rgb image to output folder.
        os.makedirs(cls._output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(cls._output_dir, 'rgb_image.png'), cls.rgb_image)
        if cls.infra1_image is not None:
            cv2.imwrite(os.path.join(cls._output_dir, 'infra1_image.png'), cls.infra1_image)
        if cls.infra2_image is not None:
            cv2.imwrite(os.path.join(cls._output_dir, 'infra2_image.png'), cls.infra2_image)

        target_description = {
            'target_type': 'charuco',
            'num_squares_height': int(NUM_SQUARE_HEIGHT),
            'num_squares_width': int(NUM_SQUARE_WIDTH),
            'longer_side_m': float(LONGER_SIDE_M),
            'marker_size_m': float(MARKER_SIZE_M),
            'num_still_frames': 1,
            'still_image_pixel_threshold': 5,
        }
        cls.target_detector = CharucoBoardDetector(target_description)

        cls.populate_pose_tree()

    @classmethod
    def populate_pose_tree(cls) -> None:
        """Populate the pose tree."""
        static_transforms = cls.static_tf_transforms
        all_children = set()
        get_parent = {}
        cls.parent_T_childs = {}

        for key, transform in static_transforms.items():
            child_frame = transform[0].child_frame_id
            parent_frame = transform[0].header.frame_id
            translation = transform[0].transform.translation
            quaternion_rotation = transform[0].transform.rotation
            # Generate transform matrix parent_T_child.
            parent_T_child = np.eye(4)
            scipy_rot_quaternion = [
                quaternion_rotation.x,
                quaternion_rotation.y,
                quaternion_rotation.z,
                quaternion_rotation.w
            ]
            parent_T_child[:3, :3] = scipy_rotation.from_quat(scipy_rot_quaternion).as_matrix()
            parent_T_child[:3, 3] = [translation.x, translation.y, translation.z]
            # We assume that the transforms are static, so we only store the first transform.
            cls.parent_T_childs[key] = parent_T_child  # Just store the first transforms.
            all_children.add(child_frame)
            if child_frame not in get_parent:
                get_parent[child_frame] = parent_frame

        # Now get every child frame w.r.t world.
        parent_frame = 'world'
        world_T_childs = {}
        for child_frame in all_children:
            if not child_frame.startswith('camera'):
                continue
            old_child_frame = child_frame
            if parent_frame != get_parent[child_frame]:
                # find transform to world.
                T = np.eye(4)
                while get_parent[child_frame] != parent_frame:
                    # iterate into the graph of poses until we get to world.
                    # TODO: check for cycles.
                    T_ = cls.parent_T_childs[f'{get_parent[child_frame]}<-{child_frame}']
                    T = T_ @ T
                    child_frame = get_parent[child_frame]
                world_T_childs[old_child_frame] = T
            else:
                world_T_childs[child_frame] = cls.parent_T_childs[f'{parent_frame}<-{child_frame}']

        # Print keys of world_T_childs.
        cls.world_T_childs = world_T_childs

        parent_frame = 'camera_1_link'
        camera_1_link_T_childs = {}
        world_T_camera_1_link = world_T_childs['camera_1_link']
        camera_1_link_T_world = np.linalg.inv(world_T_camera_1_link)
        for child_frame in all_children:
            if child_frame == parent_frame:
                continue
            world_T_child = world_T_childs[child_frame]
            camera_1_link_T_childs[child_frame] = camera_1_link_T_world @ world_T_child

        cls.camera_1_link_T_childs = camera_1_link_T_childs

    @classmethod
    def tearDownClass(cls) -> None:
        """Run once after all tests."""
        pass

    def setUp(self) -> None:
        """Run before each test."""
        pass

    def tearDown(self) -> None:
        """Run after each test."""
        pass

    @classmethod
    def save_camera_intrinsic_matrices(cls) -> None:
        # take rgb camera info and make it into a K inteinsic matrix.
        cls.K_rgb = np.array(cls.rgb_camera_info.k).reshape(3, 3)
        cls.K_infra1 = np.array(cls.infra1_camera_info.k).reshape(3, 3)
        cls.K_infra2 = np.array(cls.infra2_camera_info.k).reshape(3, 3)

    def transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Transform points that are in 3D to another frame.

        It is not homogeneous with x,y,z components.
        It then adds the extra 1 dimension to convert it to homogeneous coordinates.
        Then it transforms points from one frame to the other using T which is the 4x4
        transformation matrix. Finally it returns the transformed points in the same format
        as it was sent in.

        Args
        ----
            points (np.ndarray): Points np.ndarray N x 3.
            T (np.ndarray): Transformation matrix: 4 x 4

        Returns
        -------
            np.ndarray: Points in x, y,z in N x3

        """
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        points_transformed = T.dot(points_hom.T).T
        return points_transformed[:, :3] / points_transformed[:, [3]]

    def project_points_to_image(self, points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """
        Projects points to the image space.

        The points are in Nx3 and camera matrix is 3 x 3.
        K = np.ndarray([[fx, 0, cx ],
        [0, fy, cy],
        [0, 0, 1]])
        It returns points in N x 2. Ready to be injested into an opencv image.
        Args:
            points (np.ndarray): N x 3 points in camera frame
            camera_matrix (np.ndarray): Intrinsic matrix.

        Please be aware that this operation gets rids of scaling factor of the points in 3D, so if
        you project points into image and then back out, they will not be equivalent to the points
        you pushed into the image space.

        Returns
        -------
            np.ndarray: Points in image space

        """
        points_2D = points[:, :2] / points[:, 2, np.newaxis]
        points_2D = camera_matrix.dot(np.hstack([points_2D, np.ones((points_2D.shape[0], 1))]).T).T
        return points_2D[:, :2]

    def projectPixelsFromHomeToProjected(self,
                                         corners_2d: np.ndarray,
                                         points_in_camera_frame_from_det: np.ndarray,
                                         K_color: np.ndarray,
                                         T_color_to_infra1: np.ndarray,
                                         K_infra1: np.ndarray,
                                         image: np.ndarray,
                                         rgb_image: np.ndarray,
                                         folder_name_for_this_test: str,
                                         output_file_name: str,
                                         original_file_name: str) -> np.ndarray:
        """
        Project pixels from color to infra1.

        Args
        ----
            corners_2d: np.ndarray N, 2 - 2D pixel coordinates in color frame
            K_color: np.ndarray 3x3 - color camera intrinsic matrix
            T_color_to_infra1: np.ndarray 4x4 - transform from color to infra1
            K_infra1: np.ndarray 3x3 - infra1 camera intrinsic matrix
            image: np.ndarray - infra1 image for visualization
            output_file_name: str - output filename for visualization

        Returns
        -------
            np.ndarray: Points in image space

        """
        # First save the corners 2d on rgb image.
        for x, y in corners_2d:
            cv2.circle(rgb_image, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(f'{folder_name_for_this_test}/{original_file_name}', rgb_image)

        # project points to infra1 frame.
        points_in_infra1_frame = self.transform_points(
            points_in_camera_frame_from_det,
            T_color_to_infra1
        )

        # Now project points to infra1 frame.
        # TODO: Model distortion as well in projection and unprojection.
        projected_pixels_in_infra1_frame = self.project_points_to_image(
            points_in_infra1_frame,
            K_infra1
        )

        for x, y in projected_pixels_in_infra1_frame:
            # red for detected
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            cv2.circle(image, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(f'{folder_name_for_this_test}/{output_file_name}', image)

        return projected_pixels_in_infra1_frame

    def project_points_from_image_to_camera_frame(
        self, points_2D: np.ndarray,
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Projects points from image space (u and v) to camera frame.

        It accepts points in 2D where points are in u, v=> N x 2. And then the camera matrix
        is a K intrinsic matrix with:
        K = np.ndarray([[fx, 0, cx ],
        [0, fy, cy],
        [0, 0, 1]])
        It returns points in N x 3.

        Args
        ----
            points_2D (np.ndarray): Image pixel coordinates
            camera_matrix (np.ndarray): Intrinsi matrix

        Returns
        -------
            np.ndarray: Points in camera frame

        """
        inv_camera_matrix = np.linalg.inv(camera_matrix)
        points_3D = inv_camera_matrix.dot(
            np.hstack([points_2D, np.ones((points_2D.shape[0], 1))]).T).T
        return points_3D

    def test_quantitavely_calibration_error_rgb_to_infra1(self) -> None:
        """It behaves correctly on valid input."""
        if SKIP_TEST:
            return
        home_image = self.rgb_image.copy()
        projected_image = self.infra1_image.copy()
        projected_frame_id = 'camera_1_infra1_optical_frame'
        home_frame_id = 'camera_1_color_optical_frame'
        folder_name_for_this_test = 'quantitative_analysis_rgb_to_infra1'
        home_camera_info = self.rgb_camera_info
        projected_camera_info = self.infra1_camera_info
        home_K = self.K_rgb
        projected_camera_info_K = self.K_infra1

        os.makedirs(os.path.join(self._output_dir, folder_name_for_this_test), exist_ok=True)

        # Do 2d detection on home image.
        corners_2d_home, marker_ids_home = self.target_detector.detect_target(home_image)
        charuco_corners_home, charuco_ids_home = self.target_detector.get_corners_and_ids(
            corners_2d_home, marker_ids_home, home_image)

        # Do 2d detection on projected image.
        corners_2d_projected, marker_ids_projected = self.target_detector.detect_target(
            projected_image
        )
        ret = self.target_detector.get_corners_and_ids(
            corners_2d_projected, marker_ids_projected, projected_image)
        charuco_corners_projected, charuco_ids_projected = ret

        # first find the pose. home_T_target
        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners_home, charuco_ids_home, home_K, home_camera_info.d)

        valid_rvec_tvec_home, rvec_home, tvec_home, home_frame_T_target = ret[:4]
        obj_points_home_in_target_frame = ret[4]

        # Now find the pose in projected image. projected_T_target
        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners_projected, charuco_ids_projected,
            projected_camera_info_K, projected_camera_info.d)

        valid_rvec_tvec_projected, rvec_projected, tvec_projected = ret[:3]
        projected_frame_T_target, obj_points_projected_in_target_frame = ret[3:]

        obj_points_home = self.transform_points(obj_points_home_in_target_frame,
                                                home_frame_T_target)
        obj_points_projected = self.transform_points(
            obj_points_projected_in_target_frame,
            projected_frame_T_target
        )

        # first lets filter obj points home and obj points target to be the same points in 3d.
        # find common set of corner ids in obj points home and obj points target.
        common_corner_ids = np.intersect1d(
            charuco_ids_home.flatten(), charuco_ids_projected.flatten())

        obj_points_home_filtered = obj_points_home[common_corner_ids]
        obj_points_projected_filtered = obj_points_projected[common_corner_ids]

        # Project obj points home filtered to projected frame.
        projected_frame_T_home_frame = projected_frame_T_target @ np.linalg.inv(
            home_frame_T_target
        )
        obj_points_from_home_to_projected = self.transform_points(
            obj_points_home_filtered, projected_frame_T_home_frame)

        projected_points_home_filtered = self.project_points_to_image(
            obj_points_from_home_to_projected, projected_camera_info_K)
        projected_points_projected_filtered = self.project_points_to_image(
            obj_points_projected_filtered, projected_camera_info_K)
        black_image = np.zeros_like(home_image)
        for x, y in projected_points_home_filtered:
            # red for home
            cv2.circle(black_image, (int(round(x)), int(round(y))),
                       radius=1, color=(255, 0, 0), thickness=-1)
        for x, y in projected_points_projected_filtered:
            # blue for projected
            cv2.circle(black_image, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(self._output_dir, folder_name_for_this_test,
                                 'pnp_target_test.png'), black_image)

        # Now lets find the error in the projection.
        # projected = infra1 frame. home = rgb frame.
        error = obj_points_from_home_to_projected - obj_points_projected_filtered

        # find mean and std of error along each axis, x,y and z
        error_mean_x = np.mean(error[:, 0], axis=0)
        error_std_x = np.std(error[:, 0], axis=0)
        error_mean_y = np.mean(error[:, 1], axis=0)
        error_std_y = np.std(error[:, 1], axis=0)
        error_mean_z = np.mean(error[:, 2], axis=0)
        error_std_z = np.std(error[:, 2], axis=0)

        # This error comes from intrinsics alone.
        errors_dict = {
            'april_tag_detection_error': {
                'error_mean_x_cm': error_mean_x * 100,
                'error_std_x_cm': error_std_x * 100,
                'error_mean_y_cm': error_mean_y * 100,
                'error_std_y_cm': error_std_y * 100,
                'error_mean_z_cm': error_mean_z * 100,
                'error_std_z_cm': error_std_z * 100,
                'home_frame': home_frame_id,
                'projected_frame': projected_frame_id,
                'notes': 'This error is generated after estimating pose in both'
                         'frame, and projecting the points from home frame to'
                         'projected frame and then finding the difference in the'
                         '3d pose. We make sure to match the corner ids to make sure'
                         'we are comapring the same 3d point.'
            }
        }

        # get the transform using intrinsics only
        world_T_projected_frame = self.world_T_childs[projected_frame_id]
        world_T_home_frame = self.world_T_childs[home_frame_id]
        projected_frame_T_home_frame = np.linalg.inv(world_T_projected_frame) @ world_T_home_frame

        obj_points_from_home_to_projected = self.transform_points(
            obj_points_home_filtered,
            projected_frame_T_home_frame
        )

        # Now lets find the error in the projection.
        error = obj_points_from_home_to_projected - obj_points_projected_filtered

        # TODO: get a visualization on a black image with both points.
        projected_points_home_filtered_2 = self.project_points_to_image(
            obj_points_from_home_to_projected, projected_camera_info_K)
        projected_points_projected_filtered_2 = self.project_points_to_image(
            obj_points_projected_filtered, projected_camera_info_K)
        black_image_2 = np.zeros_like(home_image)
        for x, y in projected_points_home_filtered_2:
            # red for home
            cv2.circle(black_image_2, (int(round(x)), int(round(y))),
                       radius=1, color=(255, 0, 0), thickness=-1)
        for x, y in projected_points_projected_filtered_2:
            # blue for projected
            cv2.circle(black_image_2, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(self._output_dir, folder_name_for_this_test,
                                 'intrinsics_error_test.png'), black_image_2)

        # find mean and std of error along each axis, x,y and z
        intrinsics_error_mean_x = np.mean(error[:, 0], axis=0)
        intrinsics_error_std_x = np.std(error[:, 0], axis=0)
        intrinsics_error_mean_y = np.mean(error[:, 1], axis=0)
        intrinsics_error_std_y = np.std(error[:, 1], axis=0)
        intrinsics_error_mean_z = np.mean(error[:, 2], axis=0)
        intrinsics_error_std_z = np.std(error[:, 2], axis=0)

        errors_dict['intrinsics_error'] = {
            'error_mean_x_cm': intrinsics_error_mean_x * 100,
            'error_std_x_cm': intrinsics_error_std_x * 100,
            'error_mean_y_cm': intrinsics_error_mean_y * 100,
            'error_std_y_cm': intrinsics_error_std_y * 100,
            'error_mean_z_cm': intrinsics_error_mean_z * 100,
            'error_std_z_cm': intrinsics_error_std_z * 100,
            'projected_frame': projected_frame_id,
            'home_frame': home_frame_id,
            'notes': 'This error is the same as the above dict '
                     'however we now use the intrinsics transform stored'
                     'in factory calibration instead of using the target'
                     'pose estimation error'
        }

        if not EXTRINSICS_CALIBRATIONS:
            print('No extrinsics calibrations found, skipping extrinsics error test')
            return

        if len(EXTRINSICS_CALIBRATIONS) < 2:
            raise ValueError('EXTRINSICS_CALIBRATIONS must have at least 2 transforms')

        go_to_world_transform = EXTRINSICS_CALIBRATIONS[0]
        go_to_projected_transform = EXTRINSICS_CALIBRATIONS[1]

        def get_transform_matrix(calibration):
            rotation_dict = calibration['rotation']
            quat = [rotation_dict['x'], rotation_dict['y'], rotation_dict['z'], rotation_dict['w']]
            translation_dict = calibration['translation']
            translation = [translation_dict['x'], translation_dict['y'], translation_dict['z']]
            T = np.eye(4)
            T[:3, :3] = scipy_rotation.from_quat(quat).as_matrix()
            T[:3, 3] = translation
            return T

        camera_1_link_T_home_frame = self.camera_1_link_T_childs[home_frame_id]
        projected_frame_T_camera_1_link = np.linalg.inv(
            self.camera_1_link_T_childs[projected_frame_id]
        )

        world_T_camera_1_link_calibration_1 = get_transform_matrix(go_to_world_transform)

        world_T_camera_1_link_calibration_2 = get_transform_matrix(go_to_projected_transform)

        projected_frame_T_home_frame = projected_frame_T_camera_1_link @ \
            np.linalg.inv(world_T_camera_1_link_calibration_2) @ \
            world_T_camera_1_link_calibration_1 @ camera_1_link_T_home_frame

        obj_points_from_home_to_projected = self.transform_points(
            obj_points_home_filtered, projected_frame_T_home_frame
        )

        # Now lets find the error in the projection.
        error = obj_points_from_home_to_projected - obj_points_projected_filtered

        # TODO: get a visualization on a black image with both points.
        projected_points_home_filtered_2 = self.project_points_to_image(
            obj_points_from_home_to_projected, projected_camera_info_K)
        projected_points_projected_filtered_2 = self.project_points_to_image(
            obj_points_projected_filtered, projected_camera_info_K)
        black_image_3 = np.zeros_like(home_image)
        for x, y in projected_points_home_filtered_2:
            # red for home
            cv2.circle(black_image_3, (int(round(x)), int(round(y))),
                       radius=1, color=(255, 0, 0), thickness=-1)
        for x, y in projected_points_projected_filtered_2:
            # blue for projected
            cv2.circle(black_image_3, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(self._output_dir, folder_name_for_this_test,
                                 'extrinsics_error_test.png'), black_image_3)

        # find mean and std of error along each axis, x,y and z
        intrinsics_error_mean_x = np.mean(error[:, 0], axis=0)
        intrinsics_error_std_x = np.std(error[:, 0], axis=0)
        intrinsics_error_mean_y = np.mean(error[:, 1], axis=0)
        intrinsics_error_std_y = np.std(error[:, 1], axis=0)
        intrinsics_error_mean_z = np.mean(error[:, 2], axis=0)
        intrinsics_error_std_z = np.std(error[:, 2], axis=0)

        errors_dict['extrinsics_error'] = {
            'error_mean_x_cm': intrinsics_error_mean_x * 100,
            'error_std_x_cm': intrinsics_error_std_x * 100,
            'error_mean_y_cm': intrinsics_error_mean_y * 100,
            'error_std_y_cm': intrinsics_error_std_y * 100,
            'error_mean_z_cm': intrinsics_error_mean_z * 100,
            'error_std_z_cm': intrinsics_error_std_z * 100,
            'projected_frame': projected_frame_id,
            'home_frame': home_frame_id,
            'notes': 'This error is the same as the above dict '
                     'however we now use the extrinsics transform, 1 calibration '
                     'point to world and another one used for world back to camera_1_link'
                     ' instead of using the target'
                     'pose estimation error'
        }

        # save this errors dict in output folder.
        with open(os.path.join(
            self._output_dir,
            folder_name_for_this_test,
            'errors_dict.json'
        ), 'w') as f:
            json.dump(errors_dict, f, indent=4)

    def test_quantitavely_calibration_error_infra1_to_infra2(self) -> None:
        """It behaves correctly on valid input."""
        if SKIP_TEST:
            return
        home_image = self.infra1_image.copy()
        projected_image = self.infra2_image.copy()
        projected_frame_id = 'camera_1_infra2_optical_frame'
        home_frame_id = 'camera_1_infra1_optical_frame'
        folder_name_for_this_test = 'quantitative_analysis_infra1_to_infra2'
        home_camera_info = self.infra1_camera_info
        projected_camera_info = self.infra2_camera_info
        home_K = self.K_infra1
        projected_camera_info_K = self.K_infra2

        os.makedirs(os.path.join(self._output_dir, folder_name_for_this_test), exist_ok=True)

        # Do 2d detection on home image.
        corners_2d_home, marker_ids_home = self.target_detector.detect_target(home_image)
        charuco_corners_home, charuco_ids_home = self.target_detector.get_corners_and_ids(
            corners_2d_home, marker_ids_home, home_image)

        # Do 2d detection on projected image.
        corners_2d_projected, marker_ids_projected = self.target_detector.detect_target(
            projected_image
        )
        ret = self.target_detector.get_corners_and_ids(
            corners_2d_projected, marker_ids_projected, projected_image)
        charuco_corners_projected, charuco_ids_projected = ret
        # first find the pose. home_T_target
        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners_home, charuco_ids_home, home_K, home_camera_info.d)
        valid_rvec_tvec_home, rvec_home, tvec_home = ret[:3]
        home_frame_T_target, obj_points_home_in_target_frame = ret[3:]

        # Now find the pose in projected image. projected_T_target
        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners_projected,
            charuco_ids_projected,
            projected_camera_info_K,
            projected_camera_info.d
        )

        valid_rvec_tvec_projected, rvec_projected, tvec_projected = ret[:3]
        projected_frame_T_target, obj_points_projected_in_target_frame = ret[3:]

        obj_points_home = self.transform_points(
            obj_points_home_in_target_frame, home_frame_T_target)
        obj_points_projected = self.transform_points(
            obj_points_projected_in_target_frame, projected_frame_T_target)

        # first lets filter obj points home and obj points target to be the same points in 3d.
        # find common set of corner ids in obj points home and obj points target.
        # breakpoint()
        common_corner_ids = np.intersect1d(
            charuco_ids_home.flatten(), charuco_ids_projected.flatten())

        obj_points_home_filtered = obj_points_home[common_corner_ids]
        obj_points_projected_filtered = obj_points_projected[common_corner_ids]

        # Project obj points home filtered to projected frame.
        projected_frame_T_home_frame = projected_frame_T_target @ np.linalg.inv(
            home_frame_T_target)
        obj_points_from_home_to_projected = self.transform_points(
            obj_points_home_filtered, projected_frame_T_home_frame)

        projected_points_home_filtered = self.project_points_to_image(
            obj_points_from_home_to_projected, projected_camera_info_K)
        projected_points_projected_filtered = self.project_points_to_image(
            obj_points_projected_filtered, projected_camera_info_K)
        black_image = np.zeros_like(self.rgb_image)
        for x, y in projected_points_home_filtered:
            # red for home
            cv2.circle(black_image, (int(round(x)), int(round(y))),
                       radius=1, color=(255, 0, 0), thickness=-1)
        for x, y in projected_points_projected_filtered:
            # blue for projected
            cv2.circle(black_image, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(self._output_dir, folder_name_for_this_test,
                                 'pnp_target_test.png'), black_image)

        # Now lets find the error in the projection.
        # projected = infra1 frame. home = rgb frame.
        error = obj_points_from_home_to_projected - obj_points_projected_filtered

        # find mean and std of error along each axis, x,y and z
        error_mean_x = np.mean(error[:, 0], axis=0)
        error_std_x = np.std(error[:, 0], axis=0)
        error_mean_y = np.mean(error[:, 1], axis=0)
        error_std_y = np.std(error[:, 1], axis=0)
        error_mean_z = np.mean(error[:, 2], axis=0)
        error_std_z = np.std(error[:, 2], axis=0)

        # This error comes from intrinsics alone.
        errors_dict = {
            'april_tag_detection_error': {
                'error_mean_x_cm': error_mean_x * 100,
                'error_std_x_cm': error_std_x * 100,
                'error_mean_y_cm': error_mean_y * 100,
                'error_std_y_cm': error_std_y * 100,
                'error_mean_z_cm': error_mean_z * 100,
                'error_std_z_cm': error_std_z * 100,
                'home_frame': home_frame_id,
                'projected_frame': projected_frame_id,
                'notes': 'This error is generated after estimating pose in both'
                         'frame, and projecting the points from home frame to'
                         'projected frame and then finding the difference in the'
                         '3d pose. We make sure to match the corner ids to make sure'
                         'we are comapring the same 3d point.'
            }
        }

        # get the transform using intrinsics only
        world_T_projected_frame = self.world_T_childs[projected_frame_id]
        world_T_home_frame = self.world_T_childs[home_frame_id]
        projected_frame_T_home_frame = np.linalg.inv(world_T_projected_frame) @ world_T_home_frame

        obj_points_from_home_to_projected = self.transform_points(obj_points_home_filtered,
                                                                  projected_frame_T_home_frame)

        # Now lets find the error in the projection.
        error = obj_points_from_home_to_projected - obj_points_projected_filtered

        # TODO: get a visualization on a black image with both points.
        projected_points_home_filtered_2 = self.project_points_to_image(
            obj_points_from_home_to_projected, projected_camera_info_K)
        projected_points_projected_filtered_2 = self.project_points_to_image(
            obj_points_projected_filtered, projected_camera_info_K)
        black_image_2 = np.zeros_like(self.rgb_image)
        for x, y in projected_points_home_filtered_2:
            # red for home
            cv2.circle(black_image_2, (int(round(x)), int(round(y))),
                       radius=1, color=(255, 0, 0), thickness=-1)
        for x, y in projected_points_projected_filtered_2:
            # blue for projected
            cv2.circle(black_image_2, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(self._output_dir, folder_name_for_this_test,
                                 'intrinsics_error_test.png'), black_image_2)

        # find mean and std of error along each axis, x,y and z
        intrinsics_error_mean_x = np.mean(error[:, 0], axis=0)
        intrinsics_error_std_x = np.std(error[:, 0], axis=0)
        intrinsics_error_mean_y = np.mean(error[:, 1], axis=0)
        intrinsics_error_std_y = np.std(error[:, 1], axis=0)
        intrinsics_error_mean_z = np.mean(error[:, 2], axis=0)
        intrinsics_error_std_z = np.std(error[:, 2], axis=0)

        errors_dict['intrinsics_error'] = {
            'error_mean_x_cm': intrinsics_error_mean_x * 100,
            'error_std_x_cm': intrinsics_error_std_x * 100,
            'error_mean_y_cm': intrinsics_error_mean_y * 100,
            'error_std_y_cm': intrinsics_error_std_y * 100,
            'error_mean_z_cm': intrinsics_error_mean_z * 100,
            'error_std_z_cm': intrinsics_error_std_z * 100,
            'projected_frame': projected_frame_id,
            'home_frame': home_frame_id,
            'notes': 'This error is the same as the above dict '
                     'however we now use the intrinsics transform stored'
                     'in factory calibration instead of using the target'
                     'pose estimation error'
        }

        if not EXTRINSICS_CALIBRATIONS:
            print('No extrinsics calibrations found, skipping extrinsics error test')
            return

        if len(EXTRINSICS_CALIBRATIONS) < 2:
            raise ValueError('EXTRINSICS_CALIBRATIONS must have at least 2 transforms')

        go_to_world_transform = EXTRINSICS_CALIBRATIONS[0]
        go_to_projected_transform = EXTRINSICS_CALIBRATIONS[1]

        def get_transform_matrix(calibration):
            rotation_dict = calibration['rotation']
            quat = [rotation_dict['x'], rotation_dict['y'], rotation_dict['z'], rotation_dict['w']]
            translation_dict = calibration['translation']
            translation = [translation_dict['x'], translation_dict['y'], translation_dict['z']]
            T = np.eye(4)
            T[:3, :3] = scipy_rotation.from_quat(quat).as_matrix()
            T[:3, 3] = translation
            return T

        camera_1_link_T_home_frame = self.camera_1_link_T_childs[home_frame_id]
        projected_frame_T_camera_1_link = np.linalg.inv(
            self.camera_1_link_T_childs[projected_frame_id])
        # Now go to world using the first calibration.
        world_T_camera_1_link_calibration_1 = get_transform_matrix(go_to_world_transform)

        world_T_camera_1_link_calibration_2 = get_transform_matrix(go_to_projected_transform)

        projected_frame_T_home_frame = projected_frame_T_camera_1_link @ \
            np.linalg.inv(world_T_camera_1_link_calibration_2) @ \
            world_T_camera_1_link_calibration_1 @ \
            camera_1_link_T_home_frame

        obj_points_from_home_to_projected = self.transform_points(
            obj_points_home_filtered, projected_frame_T_home_frame)

        # Now lets find the error in the projection.
        error = obj_points_from_home_to_projected - obj_points_projected_filtered

        projected_points_home_filtered_2 = self.project_points_to_image(
            obj_points_from_home_to_projected, projected_camera_info_K)
        projected_points_projected_filtered_2 = self.project_points_to_image(
            obj_points_projected_filtered, projected_camera_info_K)
        black_image_3 = np.zeros_like(self.rgb_image)
        for x, y in projected_points_home_filtered_2:
            # red for home
            cv2.circle(black_image_3, (int(round(x)), int(round(y))),
                       radius=1, color=(255, 0, 0), thickness=-1)
        for x, y in projected_points_projected_filtered_2:
            # blue for projected
            cv2.circle(black_image_3, (int(round(x)), int(round(y))),
                       radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(self._output_dir, folder_name_for_this_test,
                                 'extrinsics_error_test.png'), black_image_3)

        # find mean and std of error along each axis, x,y and z
        intrinsics_error_mean_x = np.mean(error[:, 0], axis=0)
        intrinsics_error_std_x = np.std(error[:, 0], axis=0)
        intrinsics_error_mean_y = np.mean(error[:, 1], axis=0)
        intrinsics_error_std_y = np.std(error[:, 1], axis=0)
        intrinsics_error_mean_z = np.mean(error[:, 2], axis=0)
        intrinsics_error_std_z = np.std(error[:, 2], axis=0)

        errors_dict['extrinsics_error'] = {
            'error_mean_x_cm': intrinsics_error_mean_x * 100,
            'error_std_x_cm': intrinsics_error_std_x * 100,
            'error_mean_y_cm': intrinsics_error_mean_y * 100,
            'error_std_y_cm': intrinsics_error_std_y * 100,
            'error_mean_z_cm': intrinsics_error_mean_z * 100,
            'error_std_z_cm': intrinsics_error_std_z * 100,
            'projected_frame': projected_frame_id,
            'home_frame': home_frame_id,
            'notes': 'This error is the same as the above dict '
                     'however we now use the extrinsics transform, 1 calibration point'
                     ' to world and another one used for world back to camera_1_link'
                     ' instead of using the target'
                     'pose estimation error'
        }

        # save this errors dict in output folder.
        with open(os.path.join(
            self._output_dir,
            folder_name_for_this_test,
            'errors_dict.json'
        ), 'w') as f:
            json.dump(errors_dict, f, indent=4)

    def test_rgb_to_infra1(self) -> None:
        """It behaves correctly on valid input."""
        if SKIP_TEST:
            return
        home_image = self.rgb_image.copy()
        home_image_identifier = 'rgb_image'
        projected_image_identifier = 'infra1_image'
        projected_image = self.infra1_image.copy()
        projected_frame_id = 'camera_1_infra1_optical_frame'
        home_frame_id = 'camera_1_color_optical_frame'
        folder_name_for_this_test = 'rgb_to_infra1'
        home_camera_info = self.rgb_camera_info
        home_K = self.K_rgb
        projected_camera_info_K = self.K_infra1

        os.makedirs(os.path.join(self._output_dir, folder_name_for_this_test), exist_ok=True)

        corners_2d, marker_ids = self.target_detector.detect_target(home_image)
        plotted_image, image_with_corners = self.target_detector.plot_corners(
            home_image, corners_2d, marker_ids)

        charuco_corners, charuco_ids = self.target_detector.get_corners_and_ids(
            corners_2d, marker_ids, home_image)

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_corners.png'
            ),
            image_with_corners
        )

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_markers.png'
            ),
            plotted_image
        )

        world_T_projected_frame = self.world_T_childs[projected_frame_id]
        world_T_home_frame = self.world_T_childs[home_frame_id]
        projected_frame_T_home_frame = np.linalg.inv(world_T_projected_frame) @ world_T_home_frame

        # this is a list of 4, 2 numpy arrays
        corners_2d_np = np.stack(corners_2d, axis=0).squeeze(1).reshape(-1, 2)

        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners, charuco_ids, home_K, home_camera_info.d)
        valid_rvec_tvec, rvec, tvec, home_frame_T_target, obj_points = ret
        points_in_home_frame = self.transform_points(obj_points, home_frame_T_target)
        if not valid_rvec_tvec:
            raise ValueError('PnP failed')

        self.projectPixelsFromHomeToProjected(
            corners_2d_np,
            points_in_home_frame, home_K,
            projected_frame_T_home_frame, projected_camera_info_K,
            projected_image, home_image,
            os.path.join(self._output_dir, folder_name_for_this_test),
            f'projected_points_{home_image_identifier}_to_{projected_image_identifier}.png',
            f'points_on_detected_{home_image_identifier}.png'
        )

    def test_rgb_to_infra2(self) -> None:
        """It behaves correctly on valid input."""
        if SKIP_TEST:
            return
        home_image = self.rgb_image.copy()
        home_image_identifier = 'rgb_image'
        projected_image_identifier = 'infra2_image'
        projected_image = self.infra2_image.copy()
        projected_frame_id = 'camera_1_infra2_optical_frame'
        home_frame_id = 'camera_1_color_optical_frame'
        folder_name_for_this_test = 'rgb_to_infra2'
        home_camera_info = self.rgb_camera_info
        home_K = self.K_rgb
        projected_camera_info_K = self.K_infra2

        os.makedirs(os.path.join(self._output_dir, folder_name_for_this_test), exist_ok=True)

        corners_2d, marker_ids = self.target_detector.detect_target(home_image)
        plotted_image, image_with_corners = self.target_detector.plot_corners(
            home_image, corners_2d, marker_ids)

        charuco_corners, charuco_ids = self.target_detector.get_corners_and_ids(
            corners_2d, marker_ids, home_image)

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_corners.png'
            ),
            image_with_corners
        )

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_markers.png'
            ),
            plotted_image
        )

        world_T_projected_frame = self.world_T_childs[projected_frame_id]
        world_T_home_frame = self.world_T_childs[home_frame_id]
        projected_frame_T_home_frame = np.linalg.inv(world_T_projected_frame) @ world_T_home_frame

        # this is a list of 4, 2 numpy arrays
        corners_2d_np = np.stack(corners_2d, axis=0).squeeze(1).reshape(-1, 2)

        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners, charuco_ids, home_K, home_camera_info.d)
        valid_rvec_tvec, rvec, tvec, home_frame_T_target, obj_points = ret

        points_in_home_frame = self.transform_points(obj_points, home_frame_T_target)
        if not valid_rvec_tvec:
            raise ValueError('PnP failed')

        self.projectPixelsFromHomeToProjected(
            corners_2d_np,
            points_in_home_frame, home_K,
            projected_frame_T_home_frame, projected_camera_info_K,
            projected_image, home_image,
            os.path.join(self._output_dir, folder_name_for_this_test),
            f'projected_points_{home_image_identifier}_to_{projected_image_identifier}.png',
            f'points_on_detected_{home_image_identifier}.png'
        )

    def test_infra1_to_infra2(self) -> None:
        """It behaves correctly on valid input."""
        if SKIP_TEST:
            return
        home_image = self.infra1_image.copy()
        home_image_identifier = 'infra1_image'
        projected_image_identifier = 'infra2_image'
        projected_image = self.infra2_image.copy()
        projected_frame_id = 'camera_1_infra2_optical_frame'
        home_frame_id = 'camera_1_infra1_optical_frame'
        folder_name_for_this_test = 'infra1_to_infra2'
        home_camera_info = self.infra1_camera_info
        home_K = self.K_infra1
        projected_camera_info_K = self.K_infra2

        os.makedirs(os.path.join(self._output_dir, folder_name_for_this_test), exist_ok=True)

        corners_2d, marker_ids = self.target_detector.detect_target(home_image)
        plotted_image, image_with_corners = self.target_detector.plot_corners(
            home_image, corners_2d, marker_ids)

        charuco_corners, charuco_ids = self.target_detector.get_corners_and_ids(
            corners_2d, marker_ids, home_image)
        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_corners.png'
            ),
            image_with_corners
        )

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_markers.png'
            ),
            plotted_image
        )

        world_T_projected_frame = self.world_T_childs[projected_frame_id]
        world_T_home_frame = self.world_T_childs[home_frame_id]
        projected_frame_T_home_frame = np.linalg.inv(world_T_projected_frame) @ world_T_home_frame

        # this is a list of 4, 2 numpy arrays
        corners_2d_np = np.stack(corners_2d, axis=0).squeeze(1).reshape(-1, 2)

        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners, charuco_ids, home_K, home_camera_info.d)
        valid_rvec_tvec, rvec, tvec, home_frame_T_target, obj_points = ret

        points_in_home_frame = self.transform_points(obj_points, home_frame_T_target)
        if not valid_rvec_tvec:
            raise ValueError('PnP failed')

        self.projectPixelsFromHomeToProjected(
            corners_2d_np,
            points_in_home_frame, home_K,
            projected_frame_T_home_frame, projected_camera_info_K,
            projected_image, home_image,
            os.path.join(self._output_dir, folder_name_for_this_test),
            f'projected_points_{home_image_identifier}_to_{projected_image_identifier}.png',
            f'points_on_detected_{home_image_identifier}.png'
        )

    def test_infra2_to_infra1(self) -> None:
        """It behaves correctly on valid input."""
        if SKIP_TEST:
            return
        home_image = self.infra2_image.copy()
        home_image_identifier = 'infra2_image'
        projected_image_identifier = 'infra1_image'
        projected_image = self.infra1_image.copy()
        projected_frame_id = 'camera_1_infra1_optical_frame'
        home_frame_id = 'camera_1_infra2_optical_frame'
        folder_name_for_this_test = 'infra2_to_infra1'
        home_camera_info = self.infra2_camera_info
        home_K = self.K_infra2
        projected_camera_info_K = self.K_infra1

        os.makedirs(os.path.join(self._output_dir, folder_name_for_this_test), exist_ok=True)

        corners_2d, marker_ids = self.target_detector.detect_target(home_image)
        plotted_image, image_with_corners = self.target_detector.plot_corners(
            home_image, corners_2d, marker_ids)

        charuco_corners, charuco_ids = self.target_detector.get_corners_and_ids(
            corners_2d, marker_ids, home_image)

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_corners.png'
            ),
            image_with_corners
        )

        cv2.imwrite(
            os.path.join(
                self._output_dir, folder_name_for_this_test,
                f'{home_image_identifier}_with_markers.png'
            ),
            plotted_image
        )

        world_T_projected_frame = self.world_T_childs[projected_frame_id]
        world_T_home_frame = self.world_T_childs[home_frame_id]
        projected_frame_T_home_frame = np.linalg.inv(world_T_projected_frame) @ world_T_home_frame

        # this is a list of 4, 2 numpy arrays
        corners_2d_np = np.stack(corners_2d, axis=0).squeeze(1).reshape(-1, 2)

        ret = self.target_detector.estimate_pose_new_all_corners(
            charuco_corners, charuco_ids, home_K, home_camera_info.d)
        valid_rvec_tvec, rvec, tvec, home_frame_T_target, obj_points = ret

        points_in_home_frame = self.transform_points(obj_points, home_frame_T_target)
        if not valid_rvec_tvec:
            raise ValueError('PnP failed')

        self.projectPixelsFromHomeToProjected(
            corners_2d_np,
            points_in_home_frame, home_K,
            projected_frame_T_home_frame, projected_camera_info_K,
            projected_image, home_image,
            os.path.join(self._output_dir, folder_name_for_this_test),
            f'projected_points_{home_image_identifier}_to_{projected_image_identifier}.png',
            f'points_on_detected_{home_image_identifier}.png'
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
