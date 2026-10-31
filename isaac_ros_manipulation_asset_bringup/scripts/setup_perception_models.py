#!/usr/bin/env python3
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

"""
Isaac ROS Perception Models Setup Tool.

This script automates the download and conversion of deep learning models
required for Isaac ROS perception pipelines.
"""

import argparse
from enum import Enum
import importlib
import logging
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tarfile
from typing import List, Optional
import urllib.request

from isaac_common_py import subprocess_utils


class ModelType(Enum):
    """Supported model types for download and conversion."""

    FOUNDATIONPOSE = 'foundationpose'
    DOPE = 'dope'
    SEGMENT_ANYTHING = 'segment_anything'
    SEGMENT_ANYTHING2 = 'segment_anything2'
    MANIPULATION = 'manipulation'
    ALL = 'all'


class ModelSetup:
    """Handles the setup of perception deep learning models."""

    def __init__(
        self,
        workspace_path: str,
        verbose: bool = False,
        force: bool = False,
        foundationpose_assets: Optional[Path] = None,
        dope_assets: Optional[Path] = None,
        segment_anything_assets: Optional[Path] = None,
        segment_anything2_assets: Optional[Path] = None,
        manipulation_assets: Optional[Path] = None,
        sam_model_assets: Optional[Path] = None,
        sam2_model_assets: Optional[Path] = None
    ):
        """
        Initialize the model setup tool.

        Args
        ----
            workspace_path
                Path to the Isaac ROS workspace
            verbose
                Enable verbose output
            force
                Force re-download of models even if they already exist
            foundationpose_assets
                Path to cached FoundationPose assets
            dope_assets
                Path to cached DOPE assets
            segment_anything_assets
                Path to cached Segment Anything assets
            segment_anything2_assets
                Path to cached Segment Anything 2 assets
            manipulation_assets
                Path to cached Manipulation assets
            sam_model_assets
                Path to cached SAM model assets
            sam2_model_assets
                Path to cached SAM2 model assets

        Returns
        -------
        None

        """
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.verbose = verbose
        self.force = force
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create isaac_ros_assets and models directories if they don't exist
        self.assets_dir = self.workspace_path / 'isaac_ros_assets'
        if not self.assets_dir.exists():
            self.logger.info(f'Creating isaac_ros_assets directory at {self.assets_dir}')
            self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.assets_dir / 'models'
        if not self.models_dir.exists():
            self.logger.info(f'Creating models directory at {self.models_dir}')
            self.models_dir.mkdir(parents=True, exist_ok=True)

        self.foundationpose_assets = foundationpose_assets
        self.dope_assets = dope_assets
        self.segment_anything_assets = segment_anything_assets
        self.segment_anything2_assets = segment_anything2_assets
        self.manipulation_assets = manipulation_assets
        self.sam_model_assets = sam_model_assets
        self.sam2_model_assets = sam2_model_assets

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from a URL to the specified path.

        Args
        ----
        url
            URL to download from
        output_path
            Path to save the downloaded file

        Returns
        -------
        bool
            True if download was successful, False otherwise

        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not self.force:
            self.logger.info(f'File already exists: {output_path} - Skipping download')
            return True

        self.logger.info(f'Downloading {url} to {output_path}...')
        try:
            with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            self.logger.info(f'Download complete: {output_path}')
            return True
        except Exception as e:
            self.logger.error(f'Error downloading {url}: {e}')
            return False

    def run_command(self, command_identifier: str, cmd: List[str],
                    cwd: Optional[Path] = None) -> bool:
        """
        Run a shell command.

        Args
        ----
        command_identifier
            Identifier for the command
        cmd
            Command to run as a list of strings
        cwd
            Working directory for the command

        Returns
        -------
        bool
            True if command was successful, False otherwise

        """
        try:
            log_file = self.assets_dir / 'command.log'
            print_mode = 'all' if self.verbose else 'tail'
            subprocess_utils.run_command(
                mnemonic=command_identifier,
                command=cmd,
                log_file=log_file,
                print_mode=print_mode,
                allow_failure=False,
                cwd=cwd
            )
            return True
        except subprocess.CalledProcessError:
            return False
        except Exception as e:
            self.logger.error(f'Error executing command: {e}')
            return False

    def process_urls(
        self, base_dir: Path, urls: List[str], filename: str,
        cache_path: Optional[Path] = None
    ) -> bool:
        """
        Download and extract tarballs from URLs.

        Args
        ----
        base_dir
            Base directory to extract to
        urls
            List of URLs to download

        Returns
        -------
        bool
            True if successful, False otherwise

        """
        for url in urls:
            # Get the filename from the URL
            download_path = base_dir / filename

            # If a cache directory is provided, use the cached file if it exists
            if cache_path is not None and os.path.exists(cache_path):
                self.logger.info(f'Using cached file: {cache_path}')
                shutil.copy2(cache_path, download_path)
            else:
                # Download the file
                if not self.download_file(url, download_path):
                    return False

            # Extract the tarball
            self.logger.info(f'Extracting {download_path} to {base_dir}...')
            try:
                with tarfile.open(download_path) as tar:
                    tar.extractall(path=base_dir)
                self.logger.info('Extraction complete')

                # Remove the tarball after extraction
                download_path.unlink()
                return True
            except Exception as e:
                self.logger.error(f'Error extracting {download_path}: {e}')
                return False

        return True

    def setup_foundation_pose(self) -> bool:
        """
        Set up the FoundationPose models for pose estimation.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        # Check if foundation pose models exist
        if not (self.models_dir / 'foundationpose').exists() or \
           not (self.models_dir / 'foundationpose' / 'refine_trt_engine.plan').exists() or \
           not (self.models_dir / 'foundationpose' / 'score_trt_engine.plan').exists():
            self.logger.error('Foundation pose models not found, please check and verify.')

        # Also check if rtdetr models exists
        if not (self.models_dir / 'synthetica_detr').exists() or \
           not (self.models_dir / 'synthetica_detr' / 'sdetr_grasp.plan').exists():
            self.logger.error('Synthetica detr models not found, please check and verify.')

        self.logger.info('=== Setting up FoundationPose assets ===')
        # Check if Mac and Cheese assets already exist
        fp_dir = self.assets_dir / 'isaac_ros_foundationpose'
        mac_cheese_dir = fp_dir / 'Mac_and_cheese_0_1'

        if not fp_dir.exists() or not mac_cheese_dir.exists() or self.force:
            self.logger.info('Downloading assets for mesh and texture for foundation pose...')
            url = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/isaac/' \
                  'isaac_ros_foundationpose_assets/3.2.0/files?redirect=true' \
                  '&path=quickstart.tar.gz'
            self.process_urls(
                self.assets_dir, [url], 'quickstart.tar.gz',
                cache_path=self.foundationpose_assets
            )
        else:
            self.logger.info(f'Mac and Cheese assets already exist at {mac_cheese_dir} - '
                             'Skipping download')

        return True

    def setup_dope(self) -> bool:
        """
        Set up the DOPE model for object pose estimation.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        self.logger.info('=== Setting up DOPE model ===')

        model_dir = self.models_dir / 'dope'
        model_dir.mkdir(parents=True, exist_ok=True)
        pth_path = model_dir / 'soup_60.pth'

        if self.dope_assets is not None and os.path.exists(self.dope_assets):
            self.logger.info(f'Using cached DOPE model at {self.dope_assets}')
            shutil.copy2(self.dope_assets, pth_path)
            return True

        try:
            import gdown
        except Exception:
            self.logger.info('Installing gdown package for Google Drive downloads...')
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--break-system-packages', 'gdown'
            ])
            importlib.reload(site)
            import gdown

        # Download soup_60.pth model using gdown
        if not pth_path.exists() or self.force:
            self.logger.info('Downloading soup_60.pth DOPE model...')
            url = 'https://drive.google.com/file/d/1YlbzOpkgIisMLKMlNUAYCgeO9xhAwyrC/' \
                  'view?usp=drive_link'
            try:
                gdown.download(url, str(pth_path), fuzzy=True)
                if not pth_path.exists():
                    self.logger.error(f'Error: Failed to download {pth_path}')
                    return False
            except Exception as e:
                self.logger.error(f'Error downloading DOPE model: {e}')
                return False

        self.logger.info('DOPE model setup completed successfully')
        return True

    def setup_sam_model(self) -> bool:
        """
        Download and set up the Segment Anything (SAM) model.

        Downloads the model file, converts it to ONNX format on x86 platforms,
        and sets up the config files.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        self.logger.info('=== Setting up SAM model ===')

        # Create directories
        sam_dir = self.assets_dir / 'isaac_ros_segment_anything'
        sam_dir.mkdir(parents=True, exist_ok=True)

        models_sam_dir = self.models_dir / 'triton' / 'segment_anything' / '1'
        models_sam_dir.mkdir(parents=True, exist_ok=True)

        # Download the PTH file
        pth_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        pth_path = sam_dir / 'vit_b.pth'

        if not pth_path.exists() or self.force:
            if self.sam_model_assets is not None and os.path.exists(self.sam_model_assets):
                self.logger.info(f'Using cached SAM model at {self.sam_model_assets}')
                shutil.copy2(self.sam_model_assets, pth_path)
            else:
                self.logger.info(f'Downloading SAM model from {pth_url}...')
                if not self.download_file(pth_url, pth_path):
                    self.logger.error('Failed to download SAM model')
                    return False
                self.logger.info(f'SAM model downloaded to {pth_path}')
        else:
            self.logger.info(f'SAM model already exists at {pth_path} - Skipping download')

        onnx_path = models_sam_dir / 'model.onnx'
        # ONNX is platform-agnostic, so convert on every platform (arm64 included)
        # now that torch/onnx deps resolve natively. The TRT engine built from it
        # downstream is what is platform-specific.
        if not onnx_path.exists() or self.force:

            try:
                import segment_anything  # noqa: F401
            except ImportError:
                self.logger.error(
                    'segment_anything package not found, please install it via: '
                    'pip install git+https://github.com/facebookresearch/segment-anything.git')
                return False

            self.logger.info('Converting PyTorch model to ONNX format...')
            cmd = [
                'ros2', 'run', 'isaac_ros_segment_anything', 'torch_to_onnx.py',
                '--checkpoint', str(pth_path),
                '--output', str(onnx_path),
                '--model-type', 'vit_b',
                '--sam-type', 'SAM'
            ]

            if not self.run_command('Convert SAM to ONNX', cmd):
                self.logger.error('Failed to convert SAM model to ONNX format')
                return False
            self.logger.info(f'SAM model converted to ONNX at {onnx_path}')
        else:
            self.logger.info(f'ONNX model already exists at {onnx_path} - Skipping conversion')

        # Copy the config file
        config_src = sam_dir / 'sam_config_onnx.pbtxt'
        config_dst = self.models_dir / 'triton' / 'segment_anything' / 'config.pbtxt'

        if (not config_dst.exists() and config_src.exists()) or self.force:
            self.logger.info(f'Copying config file from {config_src} to {config_dst}')
            try:
                shutil.copy2(config_src, config_dst)
                self.logger.info('Config file copied successfully')
            except Exception as e:
                self.logger.error(f'Failed to copy config file: {e}')
                return False
        elif not config_src.exists():
            self.logger.warning(f'Config file not found at {config_src}')
        else:
            self.logger.info(f'Config file already exists at {config_dst} - Skipping copy')

        self.logger.info('SAM model setup completed successfully')
        return True

    def setup_segment_anything(self) -> bool:
        """
        Set up the Segment Anything assets and model.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        self.logger.info('=== Setting up Segment Anything assets ===')

        # Check if Segment Anything assets already exist
        sam_dir = self.assets_dir / 'isaac_ros_segment_anything'

        if not sam_dir.exists() or self.force:
            self.logger.info('Downloading assets for Segment Anything...')
            url = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/isaac/' \
                  'isaac_ros_segment_anything_assets/3.1.0/files?redirect=true' \
                  '&path=quickstart.tar.gz'
            if not self.process_urls(
                self.assets_dir, [url], 'quickstart.tar.gz',
                cache_path=self.segment_anything_assets
            ):
                return False
        else:
            self.logger.info(f'Segment Anything assets already exist at {sam_dir} - '
                             'Skipping download')

        # Set up the SAM model
        if not self.setup_sam_model():
            return False

        return True

    def setup_sam2_model(self) -> bool:
        """
        Download and set up the Segment Anything 2 (SAM2) model.

        Downloads the model file, converts it to ONNX format on x86 platforms,
        and sets up the config files and warmup data.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        self.logger.info('=== Setting up SAM2 model ===')

        # Create directories
        sam2_dir = self.assets_dir / 'isaac_ros_segment_anything2'
        sam2_dir.mkdir(parents=True, exist_ok=True)

        models_sam2_dir = self.models_dir / 'triton' / 'segment_anything2' / '1'
        models_sam2_dir.mkdir(parents=True, exist_ok=True)

        # Download the PyTorch weights from official SAM2 repo
        pth_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
        pth_path = sam2_dir / 'sam2.1_hiera_tiny.pt'

        if not pth_path.exists() or self.force:
            if self.sam2_model_assets is not None and os.path.exists(self.sam2_model_assets):
                self.logger.info(f'Using cached SAM2 model at {self.sam2_model_assets}')
                shutil.copy2(self.sam2_model_assets, pth_path)
            else:
                self.logger.info(f'Downloading SAM2 model from {pth_url}...')
                if not self.download_file(pth_url, pth_path):
                    self.logger.error('Failed to download SAM2 model')
                    return False
                self.logger.info(f'SAM2 model downloaded to {pth_path}')
        else:
            self.logger.info(f'SAM2 model already exists at {pth_path} - Skipping download')

        onnx_path = models_sam2_dir / 'model.onnx'
        if not onnx_path.exists() or self.force:
            # Install sam2 package if not already installed
            try:
                import sam2  # noqa: F401
            except ImportError:
                self.logger.info('Installing sam2 package...')
                install_cmd = [
                    sys.executable, '-m', 'pip', 'install', '--break-system-packages',
                    'git+https://github.com/facebookresearch/sam2.git'
                ]
                try:
                    subprocess.check_call(install_cmd)
                except subprocess.CalledProcessError as e:
                    self.logger.error(f'Failed to install sam2 package: {e}')
                    return False

            self.logger.info('Converting PyTorch model to ONNX format (fp16)...')
            cmd = [
                'ros2', 'run', 'isaac_ros_segment_anything2', 'sam2_onnx_exporter.py',
                '--checkpoint', str(pth_path),
                '--output', str(onnx_path),
                '--fp16'
            ]

            if not self.run_command('Convert SAM2 to ONNX', cmd):
                self.logger.error('Failed to convert SAM2 model to ONNX format')
                return False
            self.logger.info(f'SAM2 model converted to ONNX at {onnx_path}')
        else:
            self.logger.info(f'ONNX model already exists at {onnx_path} - Skipping conversion')

        # Copy the config file
        config_src = sam2_dir / 'sam2_config.pbtxt'
        config_dst = self.models_dir / 'triton' / 'segment_anything2' / 'config.pbtxt'

        if (not config_dst.exists() and config_src.exists()) or self.force:
            self.logger.info(f'Copying config file from {config_src} to {config_dst}')
            try:
                shutil.copy2(config_src, config_dst)
                self.logger.info('Config file copied successfully')
            except Exception as e:
                self.logger.error(f'Failed to copy config file: {e}')
                return False
        elif not config_src.exists():
            self.logger.warning(f'Config file not found at {config_src}')
        else:
            self.logger.info(f'Config file already exists at {config_dst} - Skipping copy')

        # Copy the warmup data
        warmup_src = sam2_dir / 'warmup'
        warmup_dst = self.models_dir / 'triton' / 'segment_anything2' / 'warmup'

        if warmup_src.exists() and (not warmup_dst.exists() or self.force):
            self.logger.info(f'Copying warmup data from {warmup_src} to {warmup_dst}')
            try:
                if warmup_dst.exists():
                    shutil.rmtree(warmup_dst)
                shutil.copytree(warmup_src, warmup_dst)
                self.logger.info('Warmup data copied successfully')
            except Exception as e:
                self.logger.error(f'Failed to copy warmup data: {e}')
                return False
        elif not warmup_src.exists():
            self.logger.warning(f'Warmup data not found at {warmup_src}')
        else:
            self.logger.info(f'Warmup data already exists at {warmup_dst} - Skipping copy')

        self.logger.info('SAM2 model setup completed successfully')
        return True

    def setup_segment_anything2(self) -> bool:
        """
        Set up the Segment Anything 2 (SAM2) assets and model.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        self.logger.info('=== Setting up Segment Anything 2 assets ===')

        # Check if Segment Anything 2 assets already exist
        sam2_dir = self.assets_dir / 'isaac_ros_segment_anything2'

        if not sam2_dir.exists() or self.force:
            self.logger.info('Downloading assets for Segment Anything 2...')
            url = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/isaac/' \
                  'isaac_ros_segment_anything2_assets/4.0.1/files?redirect=true' \
                  '&path=quickstart.tar.gz'
            if not self.process_urls(
                self.assets_dir, [url], 'quickstart.tar.gz',
                cache_path=self.segment_anything2_assets
            ):
                return False
        else:
            self.logger.info(f'Segment Anything 2 assets already exist at {sam2_dir} - '
                             'Skipping download')

        # Set up the SAM2 model
        if not self.setup_sam2_model():
            return False

        return True

    def setup_manipulation(self) -> bool:
        """
        Set up the Manipulation assets for contact-rich insertion tutorials.

        Returns
        -------
        bool
            True if setup was successful, False otherwise

        """
        self.logger.info('=== Setting up Manipulation assets ===')

        # Check if Manipulation assets already exist
        manipulation_dir = self.assets_dir / 'isaac_ros_manipulation'

        if not manipulation_dir.exists() or self.force:
            # remove the directory if it exists, in case of force download
            shutil.rmtree(manipulation_dir, ignore_errors=True)
            url = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/' \
                  'isaac/isaac_ros_manipulation_assets/5.0.4/' \
                  'files?redirect=true&path=quickstart.tar.gz'
            if not self.process_urls(
                self.assets_dir, [url], 'quickstart.tar.gz',
                cache_path=self.manipulation_assets
            ):
                return False
        else:
            self.logger.info(f'Manipulation assets already exist at {manipulation_dir} - '
                             'Skipping download')

        return True

    def print_directory_structure(self) -> None:
        """Print the directory structure of the isaac_ros_assets folder."""
        self.logger.info('=== Directory Structure ===')

        try:
            # Try to use the tree command if available
            result = subprocess.run(['which', 'tree'], capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(['tree', self.assets_dir], check=False)
            else:
                # Fallback to a simple listing
                for root, dirs, files in os.walk(self.assets_dir):
                    level = root.replace(str(self.assets_dir), '').count(os.sep)
                    indent = ' ' * 4 * level
                    self.logger.info(f'{indent}{os.path.basename(root)}/')
                    sub_indent = ' ' * 4 * (level + 1)
                    for f in files:
                        self.logger.info(f'{sub_indent}{f}')
        except Exception:
            self.logger.info(f'Could not print directory structure for {self.assets_dir}')

    def setup_models(self, model_types: List[ModelType]) -> bool:
        """
        Set up all requested models.

        Args
        ----
        model_types
            List of model types to set up

        Returns
        -------
        bool
            True if all models were set up successfully, False otherwise

        """
        success = True
        results = []

        # If ALL is selected, expand to all model types
        if ModelType.ALL in model_types:
            model_types = [m for m in ModelType if m != ModelType.ALL]

        # Process each model type
        for model_type in model_types:
            if model_type == ModelType.FOUNDATIONPOSE:
                result = self.setup_foundation_pose()
            elif model_type == ModelType.DOPE:
                result = self.setup_dope()
            elif model_type == ModelType.SEGMENT_ANYTHING:
                result = self.setup_segment_anything()
            elif model_type == ModelType.SEGMENT_ANYTHING2:
                result = self.setup_segment_anything2()
            elif model_type == ModelType.MANIPULATION:
                result = self.setup_manipulation()
            else:
                self.logger.error(f'Unknown model type: {model_type}')
                result = False

            results.append(result)
            if not result:
                success = False

        return success, results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Set up deep learning models for Isaac ROS perception pipelines'
    )
    base_path = os.getenv('ISAAC_ROS_WS')
    model_path = os.getenv('ISAAC_ROS_ASSET_MODEL_PATH')
    # If ISAAC_ROS_ASSET_MODEL_PATH is set, use it to
    # determine the workspace path.
    # This is useful when running from bazel.
    if model_path and model_path.strip():
        base_path = os.path.abspath(os.path.dirname(model_path) + '/../../../../..')
    elif base_path is None or base_path == '':
        raise ValueError(
            'Either ISAAC_ROS_ASSET_MODEL_PATH or ISAAC_ROS_WS environment variable must be set'
        )

    parser.add_argument(
        '--workspace',
        type=str,
        default=base_path,
        help='Path to the Isaac ROS workspace'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=[m.value for m in ModelType],
        default=['all'],
        help='Models to set up'
    )

    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip TensorRT conversion steps'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--show-structure',
        action='store_true',
        help='Show the directory structure after setup'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download the models'
    )

    parser.add_argument(
        '--foundationpose-assets',
        type=str,
        default=None,
        help='Path to the FoundationPose assets'
    )

    parser.add_argument(
        '--dope-assets',
        type=str,
        default=None,
        help='Path to the DOPE assets'
    )

    parser.add_argument(
        '--segment-anything-assets',
        type=str,
        default=None,
        help='Path to the Segment Anything assets'
    )

    parser.add_argument(
        '--segment-anything2-assets',
        type=str,
        default=None,
        help='Path to the Segment Anything 2 assets'
    )

    parser.add_argument(
        '--manipulation-assets',
        type=str,
        default=None,
        help='Path to the Manipulation assets'
    )

    parser.add_argument(
        '--sam-model-assets',
        type=str,
        default=None,
        help='Path to the SAM model assets'
    )

    parser.add_argument(
        '--sam2-model-assets',
        type=str,
        default=None,
        help='Path to the SAM2 model assets'
    )

    return parser.parse_args()


def main() -> int:
    """Execute the main program logic and return exit code."""
    args = parse_args()

    # Convert model type strings to enum values
    model_types = [ModelType(m) for m in args.models]

    setup = ModelSetup(
        workspace_path=args.workspace,
        verbose=args.verbose,
        force=args.force,
        foundationpose_assets=args.foundationpose_assets,
        dope_assets=args.dope_assets,
        segment_anything_assets=args.segment_anything_assets,
        segment_anything2_assets=args.segment_anything2_assets,
        manipulation_assets=args.manipulation_assets,
        sam_model_assets=args.sam_model_assets,
        sam2_model_assets=args.sam2_model_assets,
    )

    if os.environ.get('MANIPULATOR_INSTALL_ASSETS') != '1':
        setup.logger.info('Skipping model setup as MANIPULATOR_INSTALL_ASSETS is not set to 1')
        return 0

    success, results = setup.setup_models(model_types)

    if args.show_structure:
        setup.print_directory_structure()

    # Print results for each model type
    setup.logger.info('Setup results:')
    for model_type, result in zip(model_types, results):
        if model_type == ModelType.ALL:
            continue
        status = 'Success' if result else 'Failed'
        setup.logger.info(f'{status} {model_type.value}')

    if success:
        setup.logger.info('All requested models were set up successfully!')
        return 0
    else:
        setup.logger.info('Some models failed to set up. Please check the output for errors.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
