import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
import simpa as sp
import os
from collections import OrderedDict
from tmd.utils.io_iad_results import load_iad_results
from tmd.dye_analysis import DyeColors, DyeNames
from tqdm import tqdm

unmixing_wavelengths = np.arange(700, 850, 10)
target_spectrum_name = "HbO2"
oxygenation = 0
unmixing_dyes = ["B90", "BIR"]

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

if oxygenation:
    target_spectrum = torch.from_numpy(np.interp(unmixing_wavelengths, wavelengths, oxygenation * hbo2_spectrum + (1 - oxygenation) * hb_spectrum)).type(torch.float32)
abs_spectrum = load_iad_results("/home/kris/Data/Dye_project/Measured_Spectra/B93.npz")["mua"]
abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)
target_spectrum = torch.from_numpy(abs_spectrum).type(torch.float32)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"
example_spectra = os.listdir(dye_spectra_dir)

nr_of_wavelengths = len(unmixing_wavelengths)
nr_of_dyes = len(unmixing_dyes)
used_dyes = 0

chromophore_spectra_dict = OrderedDict()
input_spectra = torch.zeros([nr_of_dyes, nr_of_wavelengths])
for dye_idx, example_spectrum in enumerate(example_spectra):
    spectrum_name = example_spectrum.split(".")[0]
    if spectrum_name not in unmixing_dyes:
        continue
    c = 1#0/3 if spectrum_name != "BJ7" else 10/7
    abs_spectrum = load_iad_results(os.path.join(dye_spectra_dir, example_spectrum))["mua"]
    abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)
    chromophore_spectra_dict[spectrum_name] = c*abs_spectrum
    input_spectra[used_dyes, :] = torch.from_numpy(c*abs_spectrum).type(torch.float32)
    used_dyes += 1


class DyeConcentrationOptimizer(nn.Module):
    """Pytorch model for custom gradient optimization of dye concentrations.
    """

    def __init__(self, wavelengths, nr_of_dyes, n_iter=5000):
        super().__init__()
        self.nr_of_dyes = nr_of_dyes
        self.wavelengths = wavelengths
        # initialize weights with random numbers
        concentrations = torch.distributions.Uniform(0, 0.1).sample((nr_of_dyes,))
        # make weights torch parameters
        self.concentrations = nn.Parameter(concentrations)
        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        mixed_spectrum = torch.matmul(self.concentrations, X) #+ test_spectrum
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
            self.concentrations.data = self.concentrations.data.clamp(min=0, max=5)
            self.optimizer.zero_grad()
            losses.append(loss)
            pbar.set_description(f"loss = {loss:.2f}")
        return losses


# instantiate model
m = DyeConcentrationOptimizer(wavelengths=unmixing_wavelengths, nr_of_dyes=nr_of_dyes)
losses = m.train_loop(input_spectra=input_spectra, target_spectrum=target_spectrum)

print(m.concentrations)
conc = m.concentrations.detach()
print(f"Mixing ratio: {100*conc[0]/conc.sum():.1f}:{100*conc[1]/conc.sum():.1f}")

preds = m(input_spectra)
preds = preds.detach().numpy()
# plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.title(f"Target spectrum {target_spectrum_name}")
if target_spectrum_name == "Hb":
    c = "blue"
else:
    c = "red"
plt.semilogy(unmixing_wavelengths, target_spectrum.numpy(), label=target_spectrum_name, color=c)
plt.semilogy(unmixing_wavelengths, preds, label="Mixed spectrum", color="green")
plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
plt.legend()
plt.subplot(2, 1, 2)
for c_idx, (c_name, c_spectrum) in enumerate(chromophore_spectra_dict.items()):
    concentration = m.concentrations.detach()[c_idx]
    if concentration == 0:
        continue
    mixed_spectrum = concentration * chromophore_spectra_dict[c_name]
    plt.semilogy(unmixing_wavelengths, mixed_spectrum,
                 label=f"{c_name} ({DyeNames[c_name]}), c={concentration:.3f}",
                 color=DyeColors[c_name])
plt.legend()
plt.title(f"Endmembers")
plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
plt.tight_layout()
plt.savefig("/home/kris/Data/Dye_project/Plots/optimized_dyes.png")
