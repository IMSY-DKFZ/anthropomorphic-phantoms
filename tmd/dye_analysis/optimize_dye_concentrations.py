import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
import simpa as sp
import os
from collections import OrderedDict
from tmd.utils.io_iad_results import load_iad_results
from tqdm import tqdm

unmixing_wavelengths = np.arange(700, 850)
target_spectrum_name = "Hb"

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

target_spectra = {
    "Hb": hb_spectrum,
    "HbO2": hbo2_spectrum,
}

target_spectrum = torch.from_numpy(np.interp(unmixing_wavelengths, wavelengths, target_spectra[target_spectrum_name])).type(torch.float32)

dye_spectra_dir = "/home/kris/Work/Data/TMD/DyeSpectra/Measured_Spectra"
example_spectra = os.listdir(dye_spectra_dir)

nr_of_wavelengths = len(unmixing_wavelengths)
nr_of_dyes = len(example_spectra)

chromophore_spectra_dict = OrderedDict()
input_spectra = torch.zeros([nr_of_dyes, nr_of_wavelengths])
for dye_idx, example_spectrum in enumerate(example_spectra):
    abs_spectrum = load_iad_results(os.path.join(dye_spectra_dir, example_spectrum))["mua"]
    abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)
    chromophore_spectra_dict[example_spectrum.split(".")[0]] = abs_spectrum
    input_spectra[dye_idx, :] = torch.from_numpy(abs_spectrum).type(torch.float32)


class DyeConcentrationOptimizer(nn.Module):
    """Pytorch model for custom gradient optimization of dye concentrations.
    """

    def __init__(self, wavelengths, nr_of_dyes, n_iter=1000000):
        super().__init__()
        self.nr_of_dyes = nr_of_dyes
        self.wavelengths = wavelengths
        # initialize weights with random numbers
        concentrations = torch.distributions.Uniform(0, 0.1).sample((nr_of_dyes,))
        # make weights torch parameters
        self.concentrations = nn.Parameter(concentrations)
        self.concentrations.clamp(min=0)
        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        mixed_spectrum = torch.matmul(self.concentrations, X)
        return mixed_spectrum

    @staticmethod
    def derive(inp):
        out = inp[:-1] - inp[1:]
        return out

    def loss(self, prediction, target):
        # pred_der = self.derive(prediction)
        # target_der = self.derive(target)
        # derivative_loss = F.l1_loss(pred_der, target_der)
        # derivative_2_loss = F.l1_loss(self.derive(pred_der), self.derive(target_der))
        abs_loss = torch.sqrt(F.mse_loss(prediction, target))
        l1_loss = F.l1_loss(prediction, target)
        return abs_loss + l1_loss #+ 2*derivative_loss + 4*derivative_2_loss

    def train_loop(self, input_spectra, target_spectrum):
        losses = []
        for i in tqdm(range(self.n_iter)):
            preds = self.forward(input_spectra)
            loss = self.loss(preds, target_spectrum)
            loss.backward()
            self.optimizer.step()
            self.concentrations.data = self.concentrations.data.clamp(min=0)
            self.optimizer.zero_grad()
            losses.append(loss)
        return losses


# instantiate model
m = DyeConcentrationOptimizer(wavelengths=unmixing_wavelengths, nr_of_dyes=nr_of_dyes)
# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.001)
losses = m.train_loop(input_spectra=input_spectra, target_spectrum=target_spectrum)
plt.figure(figsize=(14, 7))

plt.plot([loss.detach() for loss in losses])
print(m.concentrations)
plt.show()
plt.close()
preds = m(input_spectra)
preds = preds.detach().numpy()
plt.figure(figsize=(14, 7))
plt.semilogy(target_spectrum.numpy())
plt.semilogy(preds)
plt.show()
plt.close()