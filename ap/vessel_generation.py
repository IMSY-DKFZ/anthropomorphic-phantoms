if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
    from simpa import Tags, Settings, VesselStructure
    import numpy as np
    np.random.seed(42)
    import time


    timer = time.time()

    _global_settings = Settings()
    _global_settings[Tags.SPACING_MM] = 1
    _global_settings[Tags.DIM_VOLUME_X_MM] = 50
    _global_settings[Tags.DIM_VOLUME_Y_MM] = 50
    _global_settings[Tags.DIM_VOLUME_Z_MM] = 50

    structure_settings = Settings()
    structure_settings[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    structure_settings[Tags.STRUCTURE_START_MM] = [25, 0, 25]
    structure_settings[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    structure_settings[Tags.STRUCTURE_RADIUS_MM] = 4
    structure_settings[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.2
    structure_settings[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    structure_settings[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = 10
    structure_settings[Tags.CONSIDER_PARTIAL_VOLUME] = True

    _global_settings.set_volume_creation_settings({Tags.SIMULATE_DEFORMED_LAYERS: False})

    vessel = VesselStructure(_global_settings, structure_settings)
    vol1 = vessel.geometrical_volume
    print("generation of the vessel took", time.time() - timer)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(vol1, shade=True)
    plt.show()