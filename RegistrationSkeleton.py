# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import time

import torch as th
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al
from airlab.loss.pairwise import _PairwiseImageLoss

from create_test_image_data import create_C_2_O_test_images


class GeneralizedLoss(_PairwiseImageLoss):

    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, size_average=True, reduce=True,
                 alpha=2.0, c=1.0):
        super(GeneralizedLoss, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._name = "Barron"
        self.warped_moving_image = None

        self.alpha = alpha
        self.c = c

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(GeneralizedLoss, self).GetCurrentMask(displacement)

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Calculate generalized and robust loss function value

        value = ...

        #########################################################################

        # mask values
        value = th.masked_select(value, mask)

        return self.return_loss(value)


def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # create test image data
    fixed_image, moving_image, shaded_image = create_C_2_O_test_images(256, dtype=dtype, device=device)

    # create image pyramide size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4], [2, 2]])
    moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4], [2, 2]])

    constant_displacement = None
    regularisation_weight = [1, 5, 50]
    number_of_iterations = [500, 500, 500]
    sigma = [[11, 11], [11, 11], [3, 3]]
    alpha = -100
    c = 0.1

    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

        registration = al.PairwiseRegistration(verbose=True)

        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Define a Bspline transformation:
        # - Provide the transform parameters (sigma and the bspline order = 3)
        # - Provide also the dtype and the device on which you want compute the transformation
        # - Set the flag diffeomorphic to True if you like

        transformation = ...

        #########################################################################

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                                  mov_im_level.size,
                                                                                  interpolation="linear")
            transformation.set_constant_flow(constant_flow)


        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Assign the transformation to the registration

        registration.set...

        #########################################################################


        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Define the implemented generalized robust loss function:
        # - Provide the two images
        # - Provide function parameters (alpha and c)
        # - Assign it as image loss to the registration

        image_loss = ...
        registration.set

        #########################################################################


        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Define regularization term on the displacement field
        # - Provide pixel spacing and regularization weight
        # - Assign it as displacement regularizer to the registration

        regulariser = ...
        registration.set...

        #########################################################################


        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Define optimizer of your choice e.g. Adam
        # - Provide transformation parameters
        # - Assign it as optimizer to the registration

        optimizer = ...
        registration.set...

        #########################################################################

        registration.set_number_of_iterations(number_of_iterations[level])


        #########################################################################
        # AirLAB - MICCAI Tutorial 2019
        #########################################################################
        # Start the registration

        ...

        #########################################################################


        constant_flow = transformation.get_flow()

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(shaded_image, displacement)
    displacement = al.create_displacement_image_from_image(displacement, moving_image)

    # create inverse displacement field
    inverse_displacement = transformation.get_inverse_displacement()
    inverse_warped_image = al.transformation.utils.warp_image(warped_image, inverse_displacement)
    inverse_displacement = al.create_displacement_image_from_image(inverse_displacement, moving_image)

    end = time.time()

if __name__ == '__main__':
    main()

