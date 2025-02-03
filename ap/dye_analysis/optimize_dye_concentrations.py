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
    Pytorch model for custom gradient optimization of dye concentrations.
    """

    def __init__(self, wavelengths, nr_of_dyes, n_iter=10000, max_concentration=5):
        super().__init__()
        self.nr_of_dyes = nr_of_dyes
        self.wavelengths = wavelengths
        # initialize weights with random numbers
        concentrations = torch.distributions.Uniform(0, 0.1).sample((nr_of_dyes,))
        # make weights torch parameters
        self.concentrations = nn.Parameter(concentrations)
        self.max_concentration = max_concentration
        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, X):
        """Implement function to be optimised. In this case, a simple matrix multiplication.,
        """
        mixed_spectrum = torch.matmul(self.concentrations, X)
        return mixed_spectrum

    @staticmethod
    def derive(inp):
        out = inp[:-1] - inp[1:]
        return out

    def loss(self, prediction, target):
        pred_der = self.derive(prediction)
        target_der = self.derive(target)
        derivative_loss = F.l1_loss(pred_der, target_der)
        derivative_2_loss = F.l1_loss(self.derive(pred_der), self.derive(target_der))
        weighted_loss = F.l1_loss(prediction, target)
        abs_loss = torch.sqrt(F.mse_loss(prediction, target))
        l1_loss = F.l1_loss(prediction, target)
        return weighted_loss#abs_loss + weighted_loss + 2*derivative_loss + 4*derivative_2_loss

    def train_loop(self, input_spectra, target_spectrum):
        """optimizing loop for minimizing given loss term.

        :param input_spectra: spectra of possible endmembers
        :param target_spectrum:
        :return: list of losses
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
    Optimize dye concentrations for a given target spectrum and input spectra.
    """
    if isinstance(target_spectrum, np.ndarray):
        target_spectrum = torch.from_numpy(target_spectrum).type(torch.float32)
    nr_of_dyes = len(input_spectra)
    # instantiate model
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
        # plt.savefig("/home/kris/Data/Dye_project/Plots/optimized_dyes.png")
        plt.show()

    return out_dict


if __name__ == "__main__":
    unmixing_wavelengths = np.arange(700, 851, 10)

    abs_spectrum = load_iad_results("/home/kris/Data/Dye_project/Measured_Spectra/B93.npz")["mua"]
    abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)

    dye_spectra_dir = "/home/kris/Data/Dye_project/publication_data/Measured_Spectra/"
    chromophore_spectra_dict = get_measured_spectra(spectra_dir=dye_spectra_dir,
                                                    unmixing_wavelengths=unmixing_wavelengths)

    _ = optimize_dye_concentrations(target_spectrum=abs_spectrum, unmixing_wavelengths=unmixing_wavelengths,
                                    input_spectra=chromophore_spectra_dict, plot_mixing_results=True)
