import json
import multiprocessing as mp
from glob import glob
from pathlib import Path
from typing import Tuple, Union, List

import matplotlib.pyplot as plt  # TODO delete
import nrrd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.ndimage import shift
from skimage.restoration import estimate_sigma


def get_energies(meta_path: Union[Path, str], number_of_files: int, number_of_wl: int, expected_id_offset: int = 0,
                 neglected_files: slice = None) -> np.ndarray:
    """ Get energies for all wavelength and all .nrrd-file of a given Scan
    :param meta_path: path to meta folder with *Info.csv containing the energy informations
    :param number_of_files: number of .nrrd-files of a given scan, needed for sanity check
    :param number_of_wl: number of wavelengths based on data, needed for sanity check
    :param expected_id_offset: expected id offset for given scan
    :param neglected files: slice indicating which files shall be neglected
    :return: energies
    """

    ''' Get and read csv file with laser energy information: '''
    if isinstance(meta_path, Path):
        meta_file = list(meta_path.glob("*Info.csv"))
    elif isinstance(meta_path, str):
        meta_file = glob(meta_path + '/*Info.csv')
    assert len(meta_file) == 1, f"No (unique) meta file in {meta_path + '_meta/*Info.csv'}"
    meta_df = pd.read_csv(meta_file[0], sep=" ", skiprows=1, names=["Idx", "sweep", "wavelength", "energy", "us"])
    energies = meta_df["energy"].to_numpy()
    sweeps = meta_df["sweep"].to_numpy()
    number_of_wl_per_sweep = np.bincount(sweeps)
    # ensure that each sweep goes through same number of wavelengths
    assert len(np.unique(number_of_wl_per_sweep)) == 1
    # ensure that IDOffset is starting at 0 otherwise the energy data mapping could go wrong
    assert meta_df["Idx"][
               0] == expected_id_offset, "IDOffset does not start as expected. Check metadata and data mapping."
    # ensure that number of wl in MetaInfo.csv file matches to number of wavelengths based on data
    assert np.unique(number_of_wl_per_sweep)[
               0] == number_of_wl, "Number of wavelengths stored in meta file does not match" \
                                   "number of wavelengths of loaded data"
    # remove energies of neglected file
    if neglected_files is not None:
        start, stop = None, None
        if neglected_files.start is not None:
            start = neglected_files.start * number_of_wl
        if neglected_files.stop is not None:
            stop = neglected_files.stop * number_of_wl
        neglected_energies = slice(start, stop)
        energies = np.delete(energies, neglected_energies)

    assert number_of_wl * number_of_files == len(energies), f'Different number of wavelength {number_of_wl} ' \
                                                            f' and files {number_of_files} compared to number of laser energy values {len(energies)}'
    return energies


def get_wavelengths(meta_path: Union[Path, str], n_wl_expected: int = None) -> np.ndarray:
    """ Get wavelengths of a given Scan
    :param meta_path: path to meta folder with '*dictionary.csv'
    :return: wavelength
    """

    # Get and read csv file with laser energy information
    if isinstance(meta_path, Path):
        meta_file = list(meta_path.glob("*dictionary.csv"))
    elif isinstance(meta_path, str):
        meta_file = glob(meta_path + '/*dictionary.csv')
    assert len(meta_file) == 1, f"No (unique) meta file in {meta_path + '_meta/*dictionary.csv'}"
    meta_df = pd.read_csv(meta_file[0], sep=",", header=0, usecols=["wavelengths_nm", "Scan"])
    readout_scan = meta_df["Scan"][0]
    wavelengths = np.array(json.loads(meta_df["wavelengths_nm"][0])).astype("int32")
    assert readout_scan.split("Scan_")[1] == (str(meta_path).split("Scan_")[1]).split("_meta")[0], "Scan number " \
                                                                                                   f"read out from *dictionary.csv {readout_scan.split('Scan_')[1]} " \
                                                                                                   f"does not fit to Scan number of input path {str(meta_path).split('Scan_')[1]}"
    if n_wl_expected is not None:
        assert n_wl_expected == len(wavelengths)
    return wavelengths


def get_mask_window(lower_limit: int, upper_limit: int, n_time_steps: int) -> np.ndarray:
    if lower_limit is not None and upper_limit is not None:
        return np.logical_and(np.arange(0, n_time_steps) >= lower_limit, np.arange(0, n_time_steps) < upper_limit)
    elif lower_limit is None and upper_limit is not None:
        return np.arange(0, n_time_steps) < upper_limit
    elif upper_limit is None and lower_limit is not None:
        return np.arange(0, n_time_steps) >= lower_limit
    else:  # both None
        return np.zeros(n_time_steps, dtype=bool)


class SignalNoiseEstimator():

    def __init__(self, device: str, signal_methods: list, noise_methods: list,
                 image_parts: list, without_outer_sensors: list, dtype: type = float,
                 plot: bool = False, compression: bool = False  # TODO delete
                 ):
        self.device = device
        self.signal_methods = signal_methods
        self.noise_methods = noise_methods
        self.image_parts = image_parts
        self.image_signal_parts = []
        self.image_noise_parts = []
        self.without_outer_sensors = without_outer_sensors
        self.dtype = dtype
        self.plot = plot  # TODO delete
        self.compression = compression
        self.signal_noise_data_collection = {
            "domain": [],
            "study": [],
            "scan": [],
            "wl": [],
            "seq": [],
            "sf": []}  # scaling factor

        for part in self.image_parts:
            if "signal" in part:
                self.image_signal_parts.append(part)
                for signal_method in self.signal_methods:
                    for wo_outer_sensors in self.without_outer_sensors:
                        key = "__".join([signal_method, part] + wo_outer_sensors * ["without_outer_sensors"])
                        self.signal_noise_data_collection[key] = []
            elif "noise" in part:
                self.image_noise_parts.append(part)
                for noise_method in self.noise_methods:
                    for wo_outer_sensors in self.without_outer_sensors:
                        key = "__".join([noise_method, part] + wo_outer_sensors * ["without_outer_sensors"])
                        self.signal_noise_data_collection[key] = []
            else:
                raise AttributeError("elements of image_parts key should containg 'signal' or 'noise'")

        """
        i.e. 
        self.signal_noise_data_collection = {
            'max__tissue_signal': [],
            'max__tissue_signal__without_outer_sensors': [],
            'mean_top10__tissue_signal': [],
            'mean_top10__tissue_signal__without_outer_sensors': [],
            'med(abs())__tissue_signal': [],
            'med(abs())__tissue_signal__without_outer_sensors': [],
            'max__skin_signal': [],
            'max__skin_signal__without_outer_sensors': [],
            'mean_top10__skin_signal': [],
            'mean_top10__skin_signal__without_outer_sensors': [],
            'med(abs())__skin_signal': [],
            'med(abs())__skin_signal__without_outer_sensors': [],
            'sigma': [],
            'sigma__noise' : [],
            'sigma__noise__without_outer_sensors': [],
            'std__noise': [],
            'std__noise__without_outer_sensors': [],
            'med(abs())__noise__without_outer_sensors': []
        }
        """

    def get_image_slice(self, part: str, without_outer_sensors: bool = True, cut_first: int = 0):
        if self.device == "REZ":
            if without_outer_sensors:
                sensor_slice = np.s_[32:256 - 32]
            else:
                sensor_slice = np.s_[:]

            if part == "noise":
                time_slice = np.s_[130 - cut_first:530 - cut_first]
            elif part == "tissue_signal":
                time_slice = np.s_[890 - cut_first:1500 - cut_first]
            elif part == "skin_signal":
                time_slice = np.s_[650:900]
        elif self.device == "REZ_in_silico":
            if without_outer_sensors:
                sensor_slice = np.s_[32:256 - 32]
            else:
                sensor_slice = np.s_[:]

            if part == "noise":
                time_slice = np.s_[130 - cut_first:530 - cut_first]
            elif part == "tissue_signal":
                time_slice = np.s_[890 - cut_first:1500 - cut_first]
            elif part == "skin_signal":
                time_slice = np.s_[650:900]
        else:
            raise AttributeError("For other devices slices are not defined yet.")

        return (sensor_slice, time_slice)

    def estimate_signal(self, ts: np.ndarray, method: str = "max"):
        if method == "max":
            # compute max over all sensors and time steps for all wl if ts is multispectral
            return np.max(ts, axis=(0, 1))  # shape (1,) or (#wl,)
        elif method == "mean_top10":
            return np.mean(np.sort(ts.flatten())[-10:])
        elif method == "med(abs())":
            # compute median of absolute values over all sensors and time steps for all wl if ts is multispectral
            return np.median(np.abs(ts), axis=(0, 1))  # shape (1,) or (#wl,)
        else:
            raise AttributeError(f"Given input for method '{method}' is not provided.")

    def estimate_noise(self, ts: np.ndarray, method: str = "sigma"):
        if method == "sigma":
            # estimate sigma over all sensors and time steps for all wl if ts is multispectral
            if ts.ndim == 2:
                return estimate_sigma(ts)  # shape (1,)
            elif ts.ndim == 3:
                return estimate_sigma(ts, channel_axis=-1)  # shape (1,) or (#wl,)
        elif method == "std":
            # compute standard deviation over all sensors and time steps for all wl if ts is multispectral
            return np.std(ts, axis=(0, 1))  # shape (1,) or (#wl,)
        elif method == "med(abs())":
            # compute median of absolute values over all sensors and time steps for all wl if ts is multispectral
            return np.median(np.abs(ts), axis=(0, 1))  # shape (1,) or (#wl,)
        else:
            raise AttributeError(f"Given input for method '{method}' is not provided.")

    def calculate_noise_estimations(self, ts: np.ndarray):
        for wo_outer_sensors in self.without_outer_sensors:
            for part in self.image_noise_parts:
                noise_slice = self.get_image_slice(part=part, without_outer_sensors=wo_outer_sensors)
                ts_noisy = ts.copy()[noise_slice]
                if self.plot:  # TODO delete
                    plt.figure(f"{part} shape={ts_noisy.shape} wo_outer_sensors={wo_outer_sensors}")  # TODO delete
                    plt.imshow(ts_noisy)  # TODO delete
                    plt.show()  # TODO delete
                for noise_method in self.noise_methods:
                    noise_est = self.estimate_noise(ts=ts_noisy, method=noise_method)
                    key = "__".join([noise_method, part] + wo_outer_sensors * ["without_outer_sensors"])
                    self.signal_noise_data_collection[key].append(noise_est.astype(self.dtype))

    def skip_noise_estimations(self):
        for wo_outer_sensors in self.without_outer_sensors:
            for part in self.image_noise_parts:
                for noise_method in self.noise_methods:
                    key = "__".join([noise_method, part] + wo_outer_sensors * ["without_outer_sensors"])
                    self.signal_noise_data_collection[key].append(np.nan)

    def calculate_signal_estimations(self, ts: np.ndarray):
        for wo_outer_sensors in self.without_outer_sensors:
            for part in self.image_signal_parts:
                tissue_signal_slice = self.get_image_slice(part=part, without_outer_sensors=wo_outer_sensors)
                ts_signal = ts.copy()[tissue_signal_slice]
                if self.plot:  # TODO delete
                    plt.figure(f"{part} shape={ts_signal.shape} wo_outer_sensors={wo_outer_sensors}")  # TODO delete
                    plt.imshow(ts_signal, aspect="auto", origin='lower')  # TODO delete
                    max_img_pos = np.unravel_index(ts_signal.argmax(), ts_signal.shape)
                    print(max_img_pos)
                    plt.scatter(max_img_pos[1], max_img_pos[0], s=80, facecolors='none', edgecolors='r')
                    plt.text(0, 0, f"{ts_signal.max():.2f}", bbox=dict(facecolor='white', alpha=0.5))
                    # if ts_signal.max()>30:
                    #    plt.show() # TODO delete
                    # else:
                    #    plt.close()
                    plt.show()
                for signal_method in self.signal_methods:
                    signal_est = self.estimate_signal(ts=ts_signal, method=signal_method)
                    key = "__".join([signal_method, part] + wo_outer_sensors * ["without_outer_sensors"])
                    self.signal_noise_data_collection[key].append(signal_est.astype(self.dtype))

    def skip_signal_estimations(self):
        for wo_outer_sensors in self.without_outer_sensors:
            for part in self.image_signal_parts:
                for signal_method in self.signal_methods:
                    key = "__".join([signal_method, part] + wo_outer_sensors * ["without_outer_sensors"])
                    self.signal_noise_data_collection[key].append(np.nan)

    def collect_meta_data(self, domain: str, study: int, scan: int, wl: int, seq: int, sf: float):
        self.signal_noise_data_collection["domain"].append(domain)
        self.signal_noise_data_collection["study"].append(study)
        self.signal_noise_data_collection["scan"].append(scan)
        self.signal_noise_data_collection["wl"].append(wl)
        self.signal_noise_data_collection["seq"].append(seq)
        self.signal_noise_data_collection["sf"].append(sf)

    def get_data_collection(self):
        return self.signal_noise_data_collection

    def store_data_collection(self, save_path: Path):
        sig_noise_data = pd.DataFrame.from_dict(self.signal_noise_data_collection)
        if self.compression:
            compression_opts = dict(method='zip', archive_name=f'{save_path.name.replace(".zip", "")}.csv')
            sig_noise_data.to_csv(save_path, index=False, compression=compression_opts)
        else:
            sig_noise_data.to_csv(save_path, index=False)

    def show_current_data_collection(self):
        for key in self.signal_noise_data_collection.keys():
            print(key, self.signal_noise_data_collection[key])


def show_noise_hist(df, keys=None, rescale=False, new_figure=True, divide_by_sf=False, plot_legend=True):
    sf = df["sf"][0]
    domain = df["domain"][0]
    print(domain, sf)
    if new_figure:
        plt.figure()
    if keys == None:
        noise_list = [df["std__noise"].copy(), df["std__noise__without_outer_sensors"].copy(),
                      df["med(abs())__noise__without_outer_sensors"].copy(), df["sigma__noise"].copy(),
                      df["sigma__noise__without_outer_sensors"].copy()]
        labels = [r"$\mathrm{std}$", r"$\mathrm{std}_{\mathrm{without\ outer\ sensors}}$",
                  r"$\mathrm{median(abs())}_{\mathrm{without\ outer\ sensors}}$", r"$\sigma$",
                  r"$\sigma_{\mathrm{without\ outer\ sensors}}$"]
        for i in range(len(noise_list)):
            xlabel = "noise metric"
            if rescale:
                noise_list[i] /= noise_list[i].mean()
                xlabel += "; divided by mean"
            if divide_by_sf and sf is not None:
                noise_list[i] /= sf
                xlabel += "; divided by C"
            sns.histplot(x=noise_list[i], label=labels[i])
            plt.xlabel(xlabel)
    else:
        for key in keys:
            values = df[key].copy()
            if rescale:
                values /= values.mean()
            if divide_by_sf and sf is not None:
                values /= sf
            print(f"{key} mean={values.mean():.3f} std={values.std():.3f}")
            sns.histplot(x=values, label=f"sf={sf}")
    if plot_legend:
        plt.legend()
    # plt.show()


def show_signal_domain_gap(signal_key, df_invivo, df_insilico0, rescale_in_vivo=True, new_figure=True, yscale="log"):
    if new_figure:
        plt.figure()
    if not rescale_in_vivo:
        plt.title("in vivo vs in silico (sf=0)")
        df_sf0 = pd.concat([df_invivo, df_insilico0])
        sns.boxplot(data=df_sf0, x="wl", y=signal_key, hue="domain", showmeans=True)
        plt.legend()
        plt.xlim(-0.5, 15.5)
        plt.yscale(yscale)
    else:
        plt.title("in vivo vs in silico (sf=0)")
        factors = []
        wl_insilico = np.unique(df_insilico0["wl"])
        for wl in wl_insilico:
            factors.append(
                df_insilico0.query('wl==@wl')[signal_key].median() / df_invivo.query('wl==@wl')[signal_key].median())
        factor = np.mean(factors)
        print(factors)
        print(factor)
        df_invivo_scaled = df_invivo.copy()
        df_invivo_scaled[signal_key] *= factor
        df_invivo_scaled["domain"] = "in_vivo_scaled"
        df_0_scaled = pd.concat([df_insilico0, df_invivo, df_invivo_scaled])
        sns.boxplot(data=df_0_scaled, x="wl", y=signal_key, hue="domain", showmeans=True)
        plt.legend()
        plt.xlim(-0.5, 15.5)
        plt.yscale(yscale)


def calculate_snr(df, signal_key="mean_top10__skin_signal__without_outer_sensors",
                  noise_key="sigma__noise__without_outer_sensors", aggregation="median"):
    wlens = np.unique(df["wl"])
    snr = []
    for wl in wlens:
        df_wl = df.query("wl == @wl")
        if aggregation == "median":
            snr.append(df_wl[signal_key].median() / df_wl[noise_key].median())
        elif aggregation == "mean":
            snr.append(df_wl[signal_key].mean() / df_wl[noise_key].mean())
    return np.array(snr)


def calculate_in_silico_snr(sf_list, df_insilico_list, signal_key="mean_top10__skin_signal__without_outer_sensors",
                            noise_key="sigma__noise__without_outer_sensors", aggregation="median"):
    snr_in_silico = {}
    for sf, df_ in zip(sf_list, df_insilico_list):
        snr_in_silico[sf] = calculate_snr(df_, signal_key, noise_key, aggregation)

    wlens = np.unique(df_insilico_list[0]["wl"])
    snr_c = {}
    for wl_idx, wl in enumerate(wlens):
        snr_c[wl] = []
        for sf in sf_list:
            snr_c[wl].append(snr_in_silico[sf][wl_idx])
        snr_c[wl] = np.array(snr_c[wl])
    return snr_c


def calculate_optimal_sf(df_invivo, df_insilico, signal_key="mean_top10__skin_signal__without_outer_sensors",
                         noise_key="sigma__noise__without_outer_sensors", wl_dependent=True):
    pass

    if wl_dependent:
        sf = {}
        for wl in np.unique(df_insilico["wl"]):
            df_invivo_wl = df_invivo.query("wl == @wl")
            df_insilico_wl = df_insilico.query("wl == @wl")
            # sf[wl] = df_insilico_wl[signal_key].mean()*df_invivo[noise_key])


class InVivoHandler():
    """
    Based on Niklas MSOT_BF scripts
    """

    def get_scans(self, DATA_PATH: str) -> list:
        """ Get list of Scans from a study / studies OR single PA file. If study / scans getData path notation is mandatory.
        :param DATA_PATH: str to study or scan.
        :return: image stack with shape [file, x, y, wavelength]
        """
        if DATA_PATH.find('.nrrd') != -1:  # if path is a single file return single file
            return [DATA_PATH]
        elif DATA_PATH.find('pc') != -1 and len(glob(DATA_PATH + '/*.nrrd')) > 0:
            # ensure '/' at the end of the file
            DATA_PATH = DATA_PATH.replace('\\', '/')
            if DATA_PATH[-1] != '/':
                DATA_PATH += '/'
            return [DATA_PATH]  # if path is a single scan that contains nrrd files return said scan

        parents = [DATA_PATH]  # kind of FIFO que to handle order in which directories are scanned
        scan_list = []  # list of scans to return
        loop_iterations = 0
        while loop_iterations < 4:  # scan for nrrd files (cf Breadth-first search / Breitensuche)
            new_parents = []  # next deeper level of parents
            for parent in parents:
                # ensure a correct path notation:
                parent = parent.replace('\\', '/')
                if parent[:-1] != '/':
                    parent += '/'
                # if a child of parent (parent + */*) contains a nrrd file it is assumed that parent is a study -
                # leveraging the getData notation the raw PA time series data are in 'study_*/Scan_*_pc/'
                # so the 'parent/*_pc' path will be added to the return list
                if len(glob(parent + '*/*.nrrd')) > 0:
                    scan_list += glob(parent + '*pc/')
                    loop_iterations = 4  # breaks loop after all current parents are scanned
                else:  # go one level deeper if no nrrd files where detected
                    new_parents.extend(glob(parent + '*'))
            parents = new_parents
            loop_iterations += 1  # ensures that loop breaks after 3 iterations
        scan_list.sort()  # sort by name
        return scan_list

    def load_file(self, file: str) -> np.ndarray:
        """ Read nrrd file without metadata
        :param file: file name of nrrd or equivalent data structure
        :return: image data as array
        """
        data, opt = nrrd.read(file)
        return data

    def load_data(self, SCAN_PATH: str, batch_size: int, batch_index: int, seq_offset: int = 0,
                  only_one_batch: bool = True):
        """ Multi processed data loading.
        :param SCAN_PATH: str to study or scan.
        :param batch_size: batch size
        :param batch_index: index indicating current batch
        :return: image stack with shape [file, x, y, wavelength]
        """
        files = []  # return list of files loaded
        if SCAN_PATH.find('.nrrd') != -1:  # if input is single file: add to files
            files.append(SCAN_PATH)
        else:
            files = glob(SCAN_PATH + '*.nrrd')  # if input is scan: add all files of scan to files

        assert len(files) > 0, f'No files in "path": {SCAN_PATH} and sub paths.'

        for i in range(len(files)):  # correct path string
            files[i] = files[i].replace('\\', '/')

        files.sort()  # sort the files (needed for non-windows systems)
        if batch_index == 0:
            print(f'  #files in scan = {len(files)}')
        if batch_size is not None:
            if (batch_index + 1) * batch_size >= len(files):
                loaded_all_batches = True
            else:
                loaded_all_batches = False
            files = files[batch_index * batch_size + seq_offset:(batch_index + 1) * batch_size + seq_offset]
            print(f'    Batch {batch_index}: #files = {len(files)}')
        else:
            batch_size = len(files)
            loaded_all_batches = True
            print(f'    Loaded all files {len(files)} in one batch')

        if only_one_batch:
            print("Process only one batch")
            loaded_all_batches = True

        # multi processed loading of data
        pool = mp.Pool(mp.cpu_count())  # get number of threads as number of cores on machine
        image_stack = pool.map(self.load_file, files)  # use function load_file with input files[i]
        pool.close()  # close threads
        pool.join()  # join threads

        # load single file to get header
        foo, opt = nrrd.read(files[0])

        # return image stack, header and list of loaded files
        return image_stack, opt, files, loaded_all_batches

    def get_meta_path(self, scan_path: str) -> str:
        """
        returns path to meta data folder

        :param scan_path: path to PA data ending with '_pc'
        :type scan_path: str
        :return: path to meta data folder
        :rtype: str
        """
        meta_path = scan_path.split('_pc')[0]
        meta_folder = meta_path + '_meta'
        return meta_folder

    def laser_correction(self, data: np.ndarray, META_PATH: str, files: list, batch_size: int, batch_index: int,
                         seq_offset: int = 0, return_energies: bool = False) -> np.ndarray:
        """ Correct for laser energy fluctuations by dividing every frame by laser output.
        :param data: PA data
        :param STUDY_PATH: path to meta data to extract energies from './*Info.csv'
        :param files: file names
        :param batch_size: batch size
        :param batch_index: index indicating current batch
        :return: data
        """

        ''' Get and read csv file with laser energy information: '''
        meta_file = glob(META_PATH + '/*Info.csv')
        assert len(meta_file) == 1, f"No (unique) meta file in {META_PATH + '/*Info.csv'}"
        meta_df = pd.read_csv(meta_file[0], sep=" ", skiprows=1, names=["Idx", "sweep", "wavelength", "energy", "us"])
        energies = meta_df["energy"].to_numpy()
        sweeps = meta_df["sweep"].to_numpy()
        number_of_wl_per_sweep = np.bincount(sweeps)
        # ensure that each sweep goes through same number of wavelengths
        assert len(np.unique(number_of_wl_per_sweep)) == 1
        # ensure that IDOffset is starting at 0 otherwise the energy data mapping could go wrong
        assert meta_df["Idx"][0] == 0, "IDOffset does not start with 0. Check metadata and data mapping."

        number_of_wl = np.shape(data)[-1]
        number_of_files = len(files)
        assert number_of_files == np.shape(data)[0], "Number of files does not match number of loaded data"
        assert np.unique(number_of_wl_per_sweep)[
                   0] == number_of_wl, "Number of wavelengths stored in meta file does not match" \
                                       "number of wavelengths of loaded data"
        if batch_size is None:
            batch_size = number_of_files  # in this case equals total number of all files within the current scan
            # check if number of PA frames ($sequences times wavelengths$) matches the number of energies within the whole scan
            assert number_of_wl * number_of_files == len(energies), f'Different number of wavelength {number_of_wl} ' \
                                                                    f' and files {number_of_files} compared to number of laser energy values {len(energies)}'

        ctr = batch_index * batch_size * number_of_wl + seq_offset * number_of_wl
        # for all sequences and all wavelengths per sequence: divide frame by energy
        for sequence in range(number_of_files):
            # check whether energy index of the first image of one sequence corresponds to the
            # correct sweep index of the sequence which is derived by reading out the file name ending
            sweep_index = int(files[sequence].split('PA.rf.')[1].split('.nrrd')[0])
            assert ctr / number_of_wl == sweep_index, f'False matching between energy index {ctr} and file {files[sequence]}'
            for img in range(number_of_wl):
                # print(files[sequence], energies[ctr]) # TODO. can be deleted, just for debugging
                data[sequence, :, :, img] = np.divide(data[sequence, :, :, img], energies[ctr])
                ctr += 1

        if not return_energies:
            # return corrected data
            return data
        else:
            return data, energies

    def average_data(self, data: np.ndarray, avg_indices: Tuple[int], num_avg_frames: int, files: list) -> Tuple[
        np.ndarray, list]:
        """ Average PA data frames per wavelength
        :param data: PA data in shape [sequence, x, y, wavelength]
        :param avg_indices: sequence window which should be used for averaging (start, stop), stop is neglected
        :param num_avg_frames: number of frames to average
        :param files: file names
        :return: averaged data, file names
        """
        # Check shape and size requirements
        data_shape = np.shape(data)
        assert len(data_shape) == 4, f'Shape: [sequence, x, y, wl] expected but got shape {data_shape}'
        assert data_shape[0] >= num_avg_frames, f'Data requires more than {data_shape[0]} sequences to average' \
                                                f' {num_avg_frames} frames.'

        # if number of frames to average matches number of sequences -> simple mean along the sequence axis
        if data_shape[0] == num_avg_frames:
            print(f"averaging {files=}")
            new_data = np.mean(data, axis=0)
            new_data = np.expand_dims(new_data, 0)  # ensure shape stays correct (sequence, x, y, wavelength)
        else:
            # do we have to crop the data?
            # crop = data_shape[0] % num_avg_frames
            # if crop:
            #     print(f'Data cropped (from back) by {crop} for averaging. Think of different averaging that sequence modulo'
            #         ' frames to average is zero')
            #     data = data[:-crop, :, :, :]
            #     files = files[:-crop]  # also crop file names!

            # # derive new shape and fill data with mean of the frames averaged
            # new_shape = list(data_shape)
            # number_of_new_seq = int(data_shape[0] / num_avg_frames)  # number of new sequences
            # new_shape[0] = number_of_new_seq
            # new_data = np.empty(new_shape)
            # for averaged_sequence in range(number_of_new_seq):
            #     new_data[averaged_sequence, :, :, :] = np.mean(data[averaged_sequence * num_avg_frames:
            #                                                         (averaged_sequence + 1) * num_avg_frames, :, :, :]
            #                                                 , axis=0)
            files = files[avg_indices[0]:avg_indices[1]]
            print(f"averaging {files=}")
            new_data = np.mean(data[avg_indices[0]:avg_indices[1], :, :, :], axis=0)[None, :, :, :]
        # return averaged data and list of file (since could have been changed by cropping)
        return new_data, files


def interpolate_broken_sensor_fixed_shift(time_series_data: np.ndarray, broken_sensor_list: List[int],
                                          neighbor_sensor_list: List[Tuple],
                                          shifts: np.ndarray = np.arange(0, 2.5, 0.5),
                                          filter_func=None, verbose=False):
    """_summary_

    :param time_series_data: _description_ sensors x time steps x wl
    :type time_series_data: np.ndarray
    :param broken_sensor: _description_
    :type broken_sensor: int
    :param shifts: _description_, defaults to np.arange(0, 2.5, 0.5)
    :type shifts: np.ndarray, optional
    :param filter_func: _description_, defaults to None
    :type filter_func: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    win = np.s_[330:-300]

    for neighbors, broken_sensor in zip(neighbor_sensor_list, broken_sensor_list):
        if verbose:
            print(f"{broken_sensor=}")
        for wl_idx in range(time_series_data.shape[-1]):
            ts_neighbors = time_series_data[neighbors, :, wl_idx]
            if filter_func is not None:
                ts_neighbors_filtered = filter_func(ts_neighbors)[:, win]  # already cropped
            else:
                ts_neighbors_filtered = ts_neighbors[:, win]
            best_shift = 0
            best_correlation = 0
            for shift_val in shifts:
                correlation = np.einsum("i,i",
                                        shift(ts_neighbors_filtered[0, :], -shift_val, mode="nearest"),
                                        shift(ts_neighbors_filtered[-1, :], shift_val, mode="nearest"))
                if correlation > best_correlation:
                    best_shift = shift_val
                    best_correlation = correlation
            if verbose:
                print(f"{best_shift=}")
            interpolation = (shift(ts_neighbors[0, :], -best_shift, mode="nearest") + shift(ts_neighbors[-1, :],
                                                                                            best_shift,
                                                                                            mode="nearest")) / 2
            time_series_data[broken_sensor, :, wl_idx] = interpolation
    return time_series_data


def correct_er_sensors(time_series_data: np.ndarray, er_sensor_list: np.ndarray = np.arange(0, 256, 8),
                       shifts: np.ndarray = np.arange(-1, 2.5, 0.5),
                       filter_func=None, verbose=False):
    """
    correct for early response sensors in time series data of CE-certified MSOT Acuity Echo.

    :param time_series_data: _description_ sensors x time steps x wl
    :type time_series_data: np.ndarray
    :param broken_sensor: _description_
    :type broken_sensor: int
    :param shifts: _description_, defaults to np.arange(0, 2.5, 0.5)
    :type shifts: np.ndarray, optional
    :param filter_func: _description_, defaults to None
    :type filter_func: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    win = np.s_[330:-600]

    ts_interpolated = interpolate_broken_sensor_fixed_shift(time_series_data=time_series_data.copy(),
                                                            broken_sensor_list=er_sensor_list[1:],
                                                            neighbor_sensor_list=list(zip(er_sensor_list[1:] + 1,
                                                                                          er_sensor_list[1:] - 1)),
                                                            shifts=np.arange(-4, 4.5, 0.5),
                                                            filter_func=filter_func,
                                                            verbose=False)

    for er_sensor in er_sensor_list:
        if verbose:
            print(f"{er_sensor=}")
        if er_sensor == 0:  # just shift by one
            time_series_data[er_sensor, 1:, :] = time_series_data[er_sensor, :-1, :]
        else:
            for wl_idx in range(time_series_data.shape[-1]):
                ts_er_sensor = np.stack(
                    [time_series_data[er_sensor, :, wl_idx].copy(), ts_interpolated[er_sensor, :, wl_idx]])
                if filter_func is not None:
                    ts_filtered_er_sensor = filter_func(ts_er_sensor)[:, win]  # already cropped
                else:
                    ts_filtered_er_sensor = ts_er_sensor[:, win]
                best_shift = 0
                best_correlation = 0
                for shift_val in shifts:
                    correlation = np.einsum("i,i",
                                            shift(ts_filtered_er_sensor[0, :], shift_val, mode="nearest"),
                                            ts_filtered_er_sensor[1, :])
                    if correlation > best_correlation:
                        best_shift = shift_val
                        best_correlation = correlation
                if verbose:
                    print(f"{best_shift=}")
                ts_data_shifted = shift(time_series_data[er_sensor, :, wl_idx], best_shift, mode="nearest")
                time_series_data[er_sensor, :, wl_idx] = ts_data_shifted
            # plt.figure()
            # plt.plot(ts_filtered_er_sensor[0])
            # plt.plot(ts_filtered_er_sensor[1])
            # plt.show()
    return time_series_data


def svd_noise_reduction(time_series_data: np.ndarray, sensor_bands: list = [0, 7], band_size: int = 32,
                        k: int = 1, filter_func=None, verbose=False):
    """
    Apply SVD techniqe for noise reduction

    :param time_series_data: sensors x time steps x wl
    :type time_series_data: np.ndarray
    :param sensor_bands: sensor bands (containing each 32 sensors) to be considered, defaults to outer bands
                         to consider all bands pass list(range(8))
    :type sensor_bands: defaults to [0,7]
    :param band_size: number of sensors within a sensor band
    :type band_size: int
    :param filter_func: _description_, defaults to None
    :type filter_func: _type_, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    """
    win = np.s_[330:-200]

    for wl_idx in range(time_series_data.shape[-1]):
        ts_wl = time_series_data[:, :, wl_idx]
        ts_noise = np.zeros_like(ts_wl)
        if filter_func is not None:
            ts_filtered = filter_func(ts_wl)
        else:
            ts_filtered = ts_wl.copy()
        for sens_band in sensor_bands:
            if verbose:
                print(f"sensor band: {sens_band}")
            u, s, vh = np.linalg.svd(ts_filtered[band_size * sens_band:band_size * (sens_band + 1), win],
                                     full_matrices=False)
            s_noise = s.copy()
            s_noise[k:] = 0
            ts_noise_band = u @ np.diag(s_noise) @ vh
            ts_noise[band_size * sens_band:band_size * (sens_band + 1), win] = ts_noise_band
        time_series_data[:, :, wl_idx] -= ts_noise
    return time_series_data


def visualize_ts_and_img_old(ts: np.ndarray, img: np.ndarray, title: str = None):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Time Series", "Reconstruction"))
    # Plot time series data
    fig.add_trace(go.Heatmap(z=ts, colorscale='Viridis'), row=1, col=1)
    # Plot reconstruction data
    fig.add_trace(go.Heatmap(z=img, colorscale='Viridis', coloraxis='coloraxis2'), row=1, col=2)
    print("her")
    # Set colorbars
    fig.update_layout(coloraxis_colorbar=dict(x=0.45, len=0.5), coloraxis2_colorbar=dict(x=0.85, len=0.5))
    # Update the figure layout
    fig.update_layout(title=title)
    # Show the figure
    fig.show()


def visualize_ts_and_img(ts: np.ndarray, img: np.ndarray, title: str = None, plotly=True):
    if plotly:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Time Series", "Reconstruction"))
        # Plot time series data
        fig.add_trace(go.Heatmap(z=ts, colorscale='Viridis'), row=1, col=1)
        # Plot reconstruction data
        fig.add_trace(go.Heatmap(z=np.rot90(img), colorscale='Viridis', coloraxis='coloraxis'), row=1, col=2)
        # Set colorbars
        fig.update_layout(coloraxis_colorbar=dict(x=0.45, len=0.5), coloraxis=dict(colorbar=dict(x=0.85, len=0.5)))
        # Update the figure layout
        fig.update_layout(title=title)
        # Show the figure
        fig.show()
    else:
        # Plot using Matplotlib
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # Plot time series data
        im0 = axs[0].imshow(ts, cmap='viridis')
        axs[0].set_title('Time Series')
        cbar0 = fig.colorbar(im0, ax=axs[0], pad=0.03, aspect=40, shrink=0.8)
        # Plot reconstruction data
        im1 = axs[1].imshow(img, cmap='viridis')
        axs[1].set_title('Reconstruction')
        cbar1 = fig.colorbar(im1, ax=axs[1], pad=0.03, aspect=40, shrink=0.8)
        # Set overall title
        fig.suptitle(title)
        # Show the plot
        plt.show()
