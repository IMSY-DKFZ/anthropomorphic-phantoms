import simpa as sp
from tmd.utils.io_iad_results import load_iad_results
import os
import numpy as np


path_to_data = "/home/kris/Data/Dye_project/Measured_Spectra"
vessel_oxy_path_dict = {
            0.0: load_iad_results(os.path.join(path_to_data, "B90.npz")),
            0.3: load_iad_results(os.path.join(path_to_data, "B93.npz")),
            0.5: load_iad_results(os.path.join(path_to_data, "B95.npz")),
            0.7: load_iad_results(os.path.join(path_to_data, "B97.npz")),
            1.0: load_iad_results(os.path.join(path_to_data, "BIR.npz")),
}

seg_dict = {
    4: 0,
    5: 0.3,
    6: 0.5,
    7: 0.7,
    8: 1
}


def get_vessel_molecule(oxygenation: float = 1.0, phantom_sos_adjustment: int = 0):
    mua = vessel_oxy_path_dict[oxygenation]["mua"]
    mus = vessel_oxy_path_dict[oxygenation]["mus"]
    wavelengths = vessel_oxy_path_dict[oxygenation]["wavelengths"]
    g = vessel_oxy_path_dict[oxygenation]["g"]

    vessel_molecule = sp.Molecule(
        name="vessel_molecule",
        absorption_spectrum=sp.Spectrum("vessel_molecule", wavelengths, mua),
        volume_fraction=1,
        scattering_spectrum=sp.Spectrum("vessel_molecule", wavelengths, mus),
        anisotropy_spectrum=sp.AnisotropySpectrumLibrary().CONSTANT_ANISOTROPY_ARBITRARY(g),
        density=1000,
        speed_of_sound=1470 + phantom_sos_adjustment,
        alpha_coefficient=1e-4,
        gruneisen_parameter=1
    )
    
    return vessel_molecule


def get_background_molecule(forearm_nr: int = 1, phantom_sos_adjustment: int = 0):
    data = load_iad_results(os.path.join(path_to_data, f"BF{forearm_nr}.npz"))
    mua = data["mua"]
    mus = data["mus"]
    wavelengths = data["wavelengths"]
    g = data["g"]

    background_molecule = sp.Molecule(
        name="background_molecule",
        absorption_spectrum=sp.Spectrum("background_molecule", wavelengths, mua),
        volume_fraction=1,
        scattering_spectrum=sp.Spectrum("background_molecule", wavelengths, mus),
        anisotropy_spectrum=sp.AnisotropySpectrumLibrary().CONSTANT_ANISOTROPY_ARBITRARY(g),
        density=838,
        speed_of_sound=1470 + phantom_sos_adjustment,
        alpha_coefficient=1e-4,
        gruneisen_parameter=1
    )

    return background_molecule


def get_bone_molecule():

    bone_absorption = -(np.log(0.8)/20) * 10

    bone_molecule = sp.Molecule(
        name="bone_molecule",
        absorption_spectrum=sp.AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(bone_absorption),
        volume_fraction=1,
        scattering_spectrum=sp.ScatteringSpectrumLibrary().CONSTANT_SCATTERING_ARBITRARY(0.1),
        anisotropy_spectrum=sp.AnisotropySpectrumLibrary().CONSTANT_ANISOTROPY_ARBITRARY(0.9),
        density=1180,
        # speed_of_sound=2424,
        speed_of_sound=1800,
        alpha_coefficient=1e-4,
        gruneisen_parameter=1
    )

    return bone_molecule


def get_air_molecule():

    air_absorption = 1e-10

    air_molecule = sp.Molecule(
        name="air_molecule",
        absorption_spectrum=sp.AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(air_absorption),
        volume_fraction=1,
        scattering_spectrum=sp.ScatteringSpectrumLibrary().CONSTANT_SCATTERING_ARBITRARY(1e-10),
        anisotropy_spectrum=sp.AnisotropySpectrumLibrary().CONSTANT_ANISOTROPY_ARBITRARY(1 - 1e-10),
        density=1000,
        speed_of_sound=343.21,
        alpha_coefficient=1e-4,
        gruneisen_parameter=1e-10
    )

    return air_molecule


def get_vessel_tissue(seg_type: int = 4, phantom_sos_adjustment: int = 0):
    return (sp.MolecularCompositionGenerator().append(get_vessel_molecule(seg_dict[seg_type], phantom_sos_adjustment))
            .get_molecular_composition(segmentation_type=seg_type))


def get_background_tissue(forearm_nr, phantom_sos_adjustment: int = 0):
    return (sp.MolecularCompositionGenerator().append(get_background_molecule(forearm_nr, phantom_sos_adjustment))
            .get_molecular_composition(segmentation_type=3))


def get_bone_tissue():
    return (sp.MolecularCompositionGenerator().append(get_bone_molecule())
            .get_molecular_composition(segmentation_type=10))


def get_air_tissue():
    return (sp.MolecularCompositionGenerator().append(get_air_molecule())
            .get_molecular_composition(segmentation_type=11))
