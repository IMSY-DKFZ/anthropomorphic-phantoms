import torch
import torch.nn as nn
from torch.functional import F
from tqdm import tqdm


class EndmemberOptimizer(nn.Module):
    """Pytorch model for custom gradient optimization of dye concentrations.
    """

    def __init__(self, wavelengths, nr_of_endmembers, filters, n_iter=5000):
        super().__init__()
        self.nr_of_endmembers = nr_of_endmembers
        self.wavelengths = wavelengths
        # initialize weights with random numbers
        e_weights = torch.distributions.Uniform(0, 0.1).sample((nr_of_endmembers,))
        # make weights torch parameters
        self.endmember_weights = nn.Parameter(e_weights)
        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.transmission_spectra = filters

    def forward(self, X):

        mixed_spectrum = torch.matmul(self.endmember_weights, X)
        measured_filter_values = torch.matmul(mixed_spectrum, self.transmission_spectra.T)
        return measured_filter_values

    def loss(self, prediction, target):

        absolute_difference = F.l1_loss(prediction, target)

        return absolute_difference

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
            self.endmember_weights.data = self.endmember_weights.data.clamp(min=0, max=5)
            self.optimizer.zero_grad()
            losses.append(loss)
            pbar.set_description(f"loss = {loss:.2f}")
        return losses



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import simpa as sp
    unmixing_wavelengths = np.arange(700, 850, 10)

    hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
        [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
    )
    wavelengths = hb_spectrum.wavelengths
    hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values
    hb_spectrum, hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum), np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)


    def gaussian(x, mu, sig):
        return (
                1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )

    trans_1 = gaussian(unmixing_wavelengths, mu=730, sig=10)
    trans_2 = gaussian(unmixing_wavelengths, mu=780, sig=10)
    trans_3 = gaussian(unmixing_wavelengths, mu=820, sig=10)

    trans_mat = np.sqrt([trans_1, trans_2, trans_3])

    optim = EndmemberOptimizer(torch.from_numpy(unmixing_wavelengths).type(torch.float32), 2, filters=torch.from_numpy(trans_mat).type(torch.float32))

    true_spectrum = 0.3*hb_spectrum + 0.7*hbo2_spectrum + np.random.normal(0, 0.2)
    measured_values = np.matmul(true_spectrum, trans_mat.T)

    input_spectra = np.stack([hb_spectrum, hbo2_spectrum])

    losses = optim.train_loop(torch.from_numpy(input_spectra).type(torch.float32), torch.from_numpy(measured_values).type(torch.float32))
    print(optim.endmember_weights)
