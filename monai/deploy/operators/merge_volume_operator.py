# Copyright 2021-2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
from typing import List

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext

from monai.transforms import ResampleToMatch
from monai.data import MetaTensor

@md.input("image1", Image, IOType.IN_MEMORY)
@md.input("image2", Image, IOType.IN_MEMORY)
@md.input("image3", Image, IOType.IN_MEMORY)
@md.output("combined_volume", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai>=1.0.1", "torch>=1.12.1", "numpy>=1.21"])
class MergeVolumeOperator(Operator):

    def __init__(self):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        image1 = op_input.get("image1")
        image2 = op_input.get("image2")
        image3 = op_input.get("image3")
        if not image1:
            raise ValueError("Input image1 is not found.")
        if not image2:
            raise ValueError("Input image2 is not found.")
        if not image3:
            raise ValueError("Input image3 is not found.")
        
        # Transpose Image arrays and set relevant metadata
        image1._metadata["affine"] = image1._metadata["nifti_affine_transform"]
        image2._metadata["affine"] = image2._metadata["nifti_affine_transform"]
        image3._metadata["affine"] = image3._metadata["nifti_affine_transform"]
        image1._data = image1._data.T
        image2._data = image2._data.T
        image3._data = image3._data.T

        # Print the shapes of the arrays and selection names
        print(f"\n{image1._metadata['selection_name']} shape: {image1.asnumpy().shape}")
        print(f"{image2._metadata['selection_name']} shape: {image2.asnumpy().shape}")
        print(f"{image3._metadata['selection_name']} shape: {image3.asnumpy().shape}")

        # Create Metatensors 
        image1_metatensor = MetaTensor(image1.asnumpy()[None], meta=image1.metadata())
        image2_metatensor = MetaTensor(image2.asnumpy()[None], meta=image2.metadata())
        image3_metatensor = MetaTensor(image3.asnumpy()[None], meta=image3.metadata())
        
        # Resample volumes
        resampler = ResampleToMatch()
        image2_resampled = resampler(image2_metatensor, image1_metatensor)
        image3_resampled = resampler(image3_metatensor, image1_metatensor)
        print(f"\n{image1._metadata['selection_name']} resampled shape: {image1_metatensor.array.shape}")
        print(f"{image2._metadata['selection_name']} resampled shape: {image2_resampled.array.shape}")
        print(f"{image3._metadata['selection_name']} resampled shape: {image3_resampled.array.shape}")

        # Create List of resampled Images
        combined_volume = [image1_metatensor, image2_resampled, image3_resampled]
        combined_volume = np.concatenate([x.array for x in combined_volume])
        combined_volume = Image(combined_volume.T, image1_metatensor.meta)
        print(f"\nCombined volume shape: {combined_volume.asnumpy().shape}\n")

        # Save the combined volume to the output context
        op_output.set(combined_volume, "combined_volume")