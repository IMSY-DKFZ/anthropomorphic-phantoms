from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from ap.dye_analysis import DyeColors, DyeNames
from tqdm import tqdm
import simpa as sp
import os
from collections import OrderedDict
from ap.utils.io_iad_results import load_iad_results
from ap.dye_analysis.measured_spectra import get_measured_spectra
plt.switch_backend("TkAgg")


class DyeConcentrationOptimizer(nn.Module):
    """
    PyTorch model for custom gradient optimization of dye concentrations.

    This module implements an optimization model to determine the optimal dye
    concentrations that best match a target spectrum when mixed with a set of
    input spectra. It uses a gradient-based approach with a custom loss function
    that can incorporate spectral derivative constraints.

    :param wavelengths: 1D tensor or array representing the wavelengths corresponding to the spectra.
    :param nr_of_dyes: Integer specifying the number of dyes (i.e., the number of concentrations to optimize).
    :param n_iter: Number of optimization iterations. Defaults to 10000.
    :param max_concentration: Maximum allowable concentration value for each dye. Defaults to 5.

    . note::
       The initial dye concentrations are randomly sampled from a uniform distribution
       between 0 and 0.1.
    """

    def __init__(self, wavelengths, nr_of_dyes, n_iter=10000, max_concentration=5):
        super().__init__()
        self.nr_of_dyes = nr_of_dyes
        self.wavelengths = wavelengths
        # Initialize dye concentrations with random values sampled uniformly between 0 and 0.1.
        concentrations = torch.distributions.Uniform(0, 0.1).sample((nr_of_dyes,))
        # Make the concentrations a learnable parameter.
        self.concentrations = nn.Parameter(concentrations)
        self.max_concentration = max_concentration
        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, X):
        """
        Compute the mixed spectrum as a weighted sum of the input spectra.

        The mixed spectrum is computed via matrix multiplication between the dye
        concentrations and the input spectra.

        :param X: Tensor of input spectra with shape (nr_of_dyes, number of wavelengths).
        :return: Tensor representing the mixed spectrum.
        """
        mixed_spectrum = torch.matmul(self.concentrations, X)
        return mixed_spectrum

    @staticmethod
    def derive(inp):
        """
        Compute the finite difference (first derivative) of the input spectrum.

        This method calculates the difference between successive elements in the input
        tensor, which approximates the derivative of the spectrum.

        :param inp: 1D tensor or array representing a spectrum.
        :return: Tensor representing the finite difference (approximate derivative) of the input.
        """
        out = inp[:-1] - inp[1:]
        return out

    def loss(self, prediction, target):
        """
        Compute the loss between the predicted and target spectra.

        The loss is computed based on the L1 loss between the predicted and target spectra.
        Additional derivative loss terms (first and second derivatives) are calculated but are
        currently not included in the final loss value. To incorporate these terms, adjust the
        returned loss accordingly.

        :param prediction: Tensor representing the predicted mixed spectrum.
        :param target: Tensor representing the target spectrum.
        :return: Scalar tensor representing the computed loss.
        """
        pred_der = self.derive(prediction)
        target_der = self.derive(target)
        derivative_loss = F.l1_loss(pred_der, target_der)
        derivative_2_loss = F.l1_loss(self.derive(pred_der), self.derive(target_der))
        weighted_loss = F.l1_loss(prediction, target)
        abs_loss = torch.sqrt(F.mse_loss(prediction, target))
        l1_loss = F.l1_loss(prediction, target)
        # Currently, only the weighted L1 loss is used.
        return weighted_loss#abs_loss + weighted_loss + 2*derivative_loss + 4*derivative_2_loss

    def train_loop(self, input_spectra, target_spectrum):
        """
        Run the optimization loop to minimize the loss between the predicted and target spectra.

        This function iteratively performs the following steps for a number of iterations specified
        by ``self.n_iter``:

          - Compute the predicted mixed spectrum using the ``forward`` method.
          - Calculate the loss between the prediction and the target spectrum using the custom loss function.
          - Perform backpropagation and update the dye concentrations using the Adam optimizer.
          - Clamp the dye concentrations to ensure they remain between 0 and ``self.max_concentration``.
          - Record the loss for monitoring the optimization process.

        :param input_spectra: Tensor of input spectra (endmembers) with shape (nr_of_dyes, number of wavelengths).
        :param target_spectrum: Tensor representing the target mixed spectrum.
        :return: List of loss values recorded at each iteration.
        """
        losses = []
        for i in (pbar := tqdm(range(self.n_iter))):
            preds = self.forward(input_spectra)
            loss = self.loss(preds, target_spectrum)
            # loss += 0.5*torch.sum(self.concentrations)
            loss.backward()
            self.optimizer.step()
            self.concentrations.data = self.concentrations.data.clamp(min=0, max=self.max_concentration)
            self.optimizer.zero_grad()
            losses.append(loss)
            pbar.set_description(f"loss = {loss:.2f}")
        return losses


def optimize_dye_concentrations(target_spectrum: np.ndarray, unmixing_wavelengths: Union[np.ndarray, list],
                                input_spectra: dict, plot_mixing_results: bool = True, n_iter: int = 10000,
                                max_concentration: int = 5):
    """
    Optimize dye concentrations to best approximate a target spectrum using a mixture of input spectra.

    This function employs the ``DyeConcentrationOptimizer`` to determine the optimal dye
    concentrations such that a weighted combination of provided input spectra matches a given
    target spectrum. The optimization is performed using gradient-based techniques over a specified
    number of iterations. Optionally, the function can display plots illustrating the target spectrum,
    the resulting mixed spectrum, and the contributions of individual dyes.

    :param target_spectrum: The target absorption spectrum to approximate. If provided as a NumPy array,
                            it is converted to a PyTorch tensor of type ``torch.float32``.
    :type target_spectrum: np.ndarray

    :param unmixing_wavelengths: A 1D array or list of wavelengths (in nm) over which the spectra are defined
                                 and to which all spectra will be interpolated.
    :type unmixing_wavelengths: Union[np.ndarray, list]

    :param input_spectra: A dictionary mapping dye names to their corresponding absorption spectra
                          (each as a NumPy array). These spectra represent the endmembers used in the mixing model.
    :type input_spectra: dict

    :param plot_mixing_results: If True, plots comparing the target spectrum with the mixed spectrum and
                                the individual contributions of dyes will be displayed. Defaults to True.
    :type plot_mixing_results: bool

    :param n_iter: The number of iterations to run the optimization loop. Defaults to 10000.
    :type n_iter: int

    :param max_concentration: The maximum allowable concentration for each dye during optimization. Defaults to 5.
    :type max_concentration: int

    :return: A dictionary mapping dye names (with non-zero optimized concentrations) to their optimized
             concentration values.
    :rtype: dict

    . note::
       This function expects the global variables ``DyeNames`` and ``DyeColors`` to be defined. These
       are used for labeling and coloring the plots when ``plot_mixing_results`` is set to True.
    """
    if isinstance(target_spectrum, np.ndarray):
        target_spectrum = torch.from_numpy(target_spectrum).type(torch.float32)
    nr_of_dyes = len(input_spectra)
    # Instantiate the model by stacking input spectra as a tensor.
    input_spectra_array = torch.stack([torch.from_numpy(spectrum).type(torch.float32) for spectrum in input_spectra.values()])

    dye_optim = DyeConcentrationOptimizer(wavelengths=unmixing_wavelengths, nr_of_dyes=nr_of_dyes,
                                          n_iter=n_iter, max_concentration=max_concentration)
    losses = dye_optim.train_loop(input_spectra=input_spectra_array, target_spectrum=target_spectrum)

    conc = dye_optim.concentrations.detach()
    non_zeros = torch.where(conc > 1e-3)[0]
    out_dict = {key: conc[idx].item() for idx, key in enumerate(input_spectra.keys()) if idx in non_zeros}
    print(f"Non-zero concentrations: ", out_dict)

    preds = dye_optim(input_spectra_array)
    if plot_mixing_results:
        preds = preds.detach().numpy()
        # plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.title(f"Target spectrum optimization")
        plt.semilogy(unmixing_wavelengths, target_spectrum.numpy(), label="Target spectrum", color="red")
        plt.semilogy(unmixing_wavelengths, preds, label="Mixed spectrum", color="green")
        plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
        plt.xlabel("Wavelength [nm]")
        plt.legend()
        plt.subplot(2, 1, 2)
        for c_idx, (c_name, c_spectrum) in enumerate(input_spectra.items()):
            concentration = dye_optim.concentrations.detach()[c_idx]
            if concentration == 0:
                continue
            mixed_spectrum = concentration * input_spectra[c_name]
            plt.semilogy(unmixing_wavelengths, mixed_spectrum,
                         label=f"{c_name} ({DyeNames[c_name]}), c={concentration:.3f}",
                         color=DyeColors[c_name])
        if len(non_zeros) <= 3:
            plt.legend()
        plt.title(f"Endmembers")
        plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
        plt.xlabel("Wavelength [nm]")
        plt.tight_layout()
        # plt.savefig("/path/to/publication_data/Plots/optimized_dyes.png")
        plt.show()

    return out_dict


if __name__ == "__main__":
    unmixing_wavelengths = np.arange(700, 851, 10)

    abs_spectrum = load_iad_results("/path/to/publication_data/Measured_Spectra/B93.npz")["mua"]
    abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)

    dye_spectra_dir = "/path/to/publication_data/Measured_Spectra/"
    chromophore_spectra_dict = get_measured_spectra(spectra_dir=dye_spectra_dir,
                                                    unmixing_wavelengths=unmixing_wavelengths)

    _ = optimize_dye_concentrations(target_spectrum=abs_spectrum, unmixing_wavelengths=unmixing_wavelengths,
                                    input_spectra=chromophore_spectra_dict, plot_mixing_results=True)
