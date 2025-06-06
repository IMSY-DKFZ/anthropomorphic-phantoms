# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F

from simpa.core.device_digital_twins.pa_devices import PhotoacousticDevice
from simpa.core.device_digital_twins.detection_geometries.curved_array import CurvedArrayDetectionGeometry
from simpa.core.device_digital_twins.illumination_geometries import IlluminationGeometryBase
from simpa.utils.settings import Settings
from simpa.utils import Tags
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
import numpy as np


class MSOTAcuityEcho(PhotoacousticDevice):
    """
    This class represents a digital twin of the MSOT Acuity Echo that was used for the acquisition of the phantom
    images.
    We need a custom class for this device because the device apparently has extreme variations in the geometry of the
    probe. The device that we use has a 3.7mm deviation of the membrane from the schematics of the device

    """

    def __init__(self, device_position_mm: np.ndarray = None,
                 field_of_view_extent_mm: np.ndarray = None):
        """
        :param device_position_mm: Each device has an internal position which serves as origin for internal \
        representations of e.g. detector element positions or illuminator positions.
        :type device_position_mm: ndarray
        :param field_of_view_extent_mm: Field of view which is defined as a numpy array of the shape \
        [xs, xe, ys, ye, zs, ze], where x, y, and z denote the coordinate axes and s and e denote the start and end \
        positions.
        :type field_of_view_extent_mm: ndarray
        """
        super(MSOTAcuityEcho, self).__init__(device_position_mm=device_position_mm,
                                             field_of_view_extent_mm=field_of_view_extent_mm)

        self.mediprene_membrane_height_mm = 1
        extended_membrane_distance = 2.8820000000000035
        self.probe_height_mm = 43.2 + extended_membrane_distance
        self.focus_in_field_of_view_mm = 8 - extended_membrane_distance
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0, self.focus_in_field_of_view_mm]))

        if field_of_view_extent_mm is None:
            self.field_of_view_extent_mm = np.asarray([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                       (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                       0, 0, 0, 50])
        else:
            self.field_of_view_extent_mm = field_of_view_extent_mm

        detection_geometry_fov = self.field_of_view_extent_mm.copy().astype(float)
        detection_geometry_fov[4] -= self.focus_in_field_of_view_mm
        detection_geometry_fov[5] -= self.focus_in_field_of_view_mm

        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=self.detection_geometry_position_vector,
                                                          field_of_view_extent_mm=detection_geometry_fov)

        self.set_detection_geometry(detection_geometry)
        illumination_geometry = MSOTAcuityIlluminationGeometry()

        # y position relative to the membrane:
        # The laser is located 43.2 + extended_membrane_distance mm  behind the membrane with an angle of 22.4 degrees.
        # However, the incident of laser and image plane is located 2.8 behind the membrane (outside of the device).
        y_pos_relative_to_membrane = np.tan(np.deg2rad(22.4)) * (43.2 + 2.8)
        self.add_illumination_geometry(illumination_geometry,
                                       illuminator_position_relative_to_pa_device=np.array([0,
                                                                                            -y_pos_relative_to_membrane,
                                                                                            -43.2 + extended_membrane_distance]))

    def update_settings_for_use_of_model_based_volume_creator(self, global_settings: Settings):
        """
        Updates the volume creation settings of the model based volume creator according to the size of the device.
        :param global_settings: Settings for the entire simulation pipeline.
        :type global_settings: Settings
        """
        try:
            volume_creator_settings = Settings(global_settings.get_volume_creation_settings())
        except KeyError as e:
            self.logger.warning("You called the update_settings_for_use_of_model_based_volume_creator method "
                                "even though there are no volume creation settings defined in the "
                                "settings dictionary.")
            return

        probe_size_mm = self.probe_height_mm
        mediprene_layer_height_mm = self.mediprene_membrane_height_mm
        heavy_water_layer_height_mm = probe_size_mm - mediprene_layer_height_mm
        spacing_mm = global_settings[Tags.SPACING_MM]
        old_volume_height_pixels = round(global_settings[Tags.DIM_VOLUME_Z_MM] / spacing_mm)

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_thickness = np.random.normal(0.4, 0.1)
        else:
            us_gel_thickness = 0

        z_dim_position_shift_mm = mediprene_layer_height_mm + heavy_water_layer_height_mm + us_gel_thickness

        new_volume_height_mm = global_settings[Tags.DIM_VOLUME_Z_MM] + z_dim_position_shift_mm

        # adjust the z-dim to msot probe height
        global_settings[Tags.DIM_VOLUME_Z_MM] = new_volume_height_mm

        # adjust the x-dim to msot probe width
        # 1 voxel is added (0.5 on both sides) to make sure no rounding errors lead to a detector element being outside
        # of the simulated volume.

        if global_settings[Tags.DIM_VOLUME_X_MM] < round(self.detection_geometry.probe_width_mm) + spacing_mm:
            width_shift_for_structures_mm = (round(self.detection_geometry.probe_width_mm) + spacing_mm -
                                             global_settings[Tags.DIM_VOLUME_X_MM]) / 2
            global_settings[Tags.DIM_VOLUME_X_MM] = round(self.detection_geometry.probe_width_mm) + spacing_mm
            self.logger.debug(f"Changed Tags.DIM_VOLUME_X_MM to {global_settings[Tags.DIM_VOLUME_X_MM]}")
        else:
            width_shift_for_structures_mm = 0

        self.logger.debug(volume_creator_settings)

        for structure_key in volume_creator_settings[Tags.STRUCTURES]:
            self.logger.debug("Adjusting " + str(structure_key))
            structure_dict = volume_creator_settings[Tags.STRUCTURES][structure_key]
            if Tags.STRUCTURE_START_MM in structure_dict:
                for molecule in structure_dict[Tags.MOLECULE_COMPOSITION]:
                    old_volume_fraction = getattr(molecule, "volume_fraction")
                    if isinstance(old_volume_fraction, torch.Tensor):
                        if old_volume_fraction.shape[2] == old_volume_height_pixels:
                            width_shift_pixels = round(width_shift_for_structures_mm / spacing_mm)
                            z_shift_pixels = round(z_dim_position_shift_mm / spacing_mm)
                            padding_height = (z_shift_pixels, 0, 0, 0, 0, 0)
                            padding_width = ((width_shift_pixels, width_shift_pixels), (0, 0), (0, 0))
                            padded_up = F.pad(old_volume_fraction, padding_height, mode='constant', value=0)
                            padded_vol = np.pad(padded_up.numpy(), padding_width, mode='edge')
                            setattr(molecule, "volume_fraction", torch.from_numpy(padded_vol))
                structure_dict[Tags.STRUCTURE_START_MM][0] = structure_dict[Tags.STRUCTURE_START_MM][
                    0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_START_MM][2] = structure_dict[Tags.STRUCTURE_START_MM][
                    2] + z_dim_position_shift_mm
            if Tags.STRUCTURE_END_MM in structure_dict:
                structure_dict[Tags.STRUCTURE_END_MM][0] = structure_dict[Tags.STRUCTURE_END_MM][
                    0] + width_shift_for_structures_mm
                structure_dict[Tags.STRUCTURE_END_MM][2] = structure_dict[Tags.STRUCTURE_END_MM][
                    2] + z_dim_position_shift_mm

        if Tags.CONSIDER_PARTIAL_VOLUME_IN_DEVICE in volume_creator_settings:
            consider_partial_volume = volume_creator_settings[Tags.CONSIDER_PARTIAL_VOLUME_IN_DEVICE]
        else:
            consider_partial_volume = False

        if Tags.US_GEL in volume_creator_settings and volume_creator_settings[Tags.US_GEL]:
            us_gel_layer_settings = Settings({
                Tags.PRIORITY: 5,
                Tags.STRUCTURE_START_MM: [0, 0,
                                          heavy_water_layer_height_mm + mediprene_layer_height_mm],
                Tags.STRUCTURE_END_MM: [0, 0,
                                        heavy_water_layer_height_mm + mediprene_layer_height_mm + us_gel_thickness],
                Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
                Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.ultrasound_gel(),
                Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
            })

            volume_creator_settings[Tags.STRUCTURES]["us_gel"] = us_gel_layer_settings

        mediprene_layer_settings = Settings({
            Tags.PRIORITY: 5,
            Tags.STRUCTURE_START_MM: [0, 0, heavy_water_layer_height_mm],
            Tags.STRUCTURE_END_MM: [0, 0, heavy_water_layer_height_mm + mediprene_layer_height_mm],
            Tags.CONSIDER_PARTIAL_VOLUME: consider_partial_volume,
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.mediprene(),
            Tags.STRUCTURE_TYPE: Tags.HORIZONTAL_LAYER_STRUCTURE
        })

        volume_creator_settings[Tags.STRUCTURES]["mediprene"] = mediprene_layer_settings

        self.device_position_mm = np.add(self.device_position_mm, np.array([width_shift_for_structures_mm, 0,
                                                                            probe_size_mm]))
        self.detection_geometry_position_vector = np.add(self.device_position_mm,
                                                         np.array([0, 0,
                                                                   self.focus_in_field_of_view_mm]))
        detection_geometry = CurvedArrayDetectionGeometry(pitch_mm=0.34,
                                                          radius_mm=40,
                                                          number_detector_elements=256,
                                                          detector_element_width_mm=0.24,
                                                          detector_element_length_mm=13,
                                                          center_frequency_hz=3.96e6,
                                                          bandwidth_percent=55,
                                                          sampling_frequency_mhz=40,
                                                          angular_origin_offset=np.pi,
                                                          device_position_mm=self.detection_geometry_position_vector,
                                                          field_of_view_extent_mm=self.field_of_view_extent_mm)

        self.set_detection_geometry(detection_geometry)
        for illumination_geom in self.illumination_geometries:
            illumination_geom.device_position_mm = np.add(illumination_geom.device_position_mm,
                                                          np.array([width_shift_for_structures_mm, 0, probe_size_mm]))

        background_settings = Settings({
            Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.heavy_water(),
            Tags.STRUCTURE_TYPE: Tags.BACKGROUND
        })
        volume_creator_settings[Tags.STRUCTURES][Tags.BACKGROUND] = background_settings

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        device_dict = {"MSOTAcuityEcho": serialized_device}
        return device_dict

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTAcuityEcho()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device


class MSOTAcuityIlluminationGeometry(IlluminationGeometryBase):
    """
    This class represents the illumination geometry of the MSOT Acuity (Echo) photoacoustic device.
    The position is defined as the middle of the illumination slit.
    """

    def __init__(self):
        """
        Initializes the illumination source.
        """
        super().__init__()

        # y position relative to the membrane:
        # The laser is located 43.2 + extended_membrane_distance mm  behind the membrane with an angle of 22.4 degrees.
        # However, the incident of laser and image plane is located 2.8 behind the membrane (outside of the device).
        y_pos_relative_to_membrane = np.tan(np.deg2rad(22.4)) * (43.2 + 2.8)

        direction_vector = np.array([0, y_pos_relative_to_membrane, 43.2 + 2.8])
        self.source_direction_vector = direction_vector/np.linalg.norm(direction_vector)
        self.normalized_source_direction_vector = self.source_direction_vector / np.linalg.norm(
            self.source_direction_vector)

        divergence_angle = 8.66  # full beam divergence angle measured at Full Width at Half Maximum (FWHM)
        full_width_at_half_maximum = 2.0 * np.tan(0.5 * np.deg2rad(divergence_angle))  # FWHM of beam divergence
        # standard deviation of gaussian with FWHM
        self.sigma = full_width_at_half_maximum / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def get_mcx_illuminator_definition(self, global_settings: Settings):
        source_type = Tags.ILLUMINATION_TYPE_SLIT
        spacing = global_settings[Tags.SPACING_MM]
        device_position = list(self.device_position_mm / spacing + 0.5)
        device_length = 30 / spacing
        source_pos = device_position
        source_pos[0] -= 0.5 * device_length
        source_direction = list(self.normalized_source_direction_vector)
        source_param1 = [device_length, 0.0, 0.0, 0.0]
        source_param2 = [self.sigma, self.sigma, 0.0, 0.0]

        return {
            "Type": source_type,
            "Pos": source_pos,
            "Dir": source_direction,
            "Param1": source_param1,
            "Param2": source_param2
        }

    def serialize(self) -> dict:
        serialized_device = self.__dict__
        return {"MSOTAcuityIlluminationGeometry": serialized_device}

    @staticmethod
    def deserialize(dictionary_to_deserialize):
        deserialized_device = MSOTAcuityIlluminationGeometry()
        for key, value in dictionary_to_deserialize.items():
            deserialized_device.__dict__[key] = value
        return deserialized_device
