from tmd.utils.io_iad_results import load_iad_results
from tmd.dye_analysis import DyeColors, DyeNames
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os

unmixing_wavelengths = np.arange(700, 855, 10)

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"

spectrum_B30 = load_iad_results(os.path.join(dye_spectra_dir, "B30.npz"))["mua"]
spectrum_B30 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B30)

spectrum_B43 = load_iad_results(os.path.join(dye_spectra_dir, "B43.npz"))["mua"]
spectrum_B43 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B43)

spectrum_B42 = load_iad_results(os.path.join(dye_spectra_dir, "B42.npz"))["mua"]
spectrum_B42 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B42)

spectrum_B66 = load_iad_results(os.path.join(dye_spectra_dir, "B66.npz"))["mua"]
spectrum_B66 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B66)

spectrum_B67 = load_iad_results(os.path.join(dye_spectra_dir, "B67.npz"))["mua"]
spectrum_B67 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B67)

spectrum_B90 = load_iad_results(os.path.join(dye_spectra_dir, "B90.npz"))["mua"]
spectrum_B90 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B90)

spectrum_B92 = load_iad_results(os.path.join(dye_spectra_dir, "B92.npz"))["mua"]
spectrum_B92 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B92)

spectrum_B93 = load_iad_results(os.path.join(dye_spectra_dir, "B93.npz"))["mua"]
spectrum_B93 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B93)

spectrum_B95 = load_iad_results(os.path.join(dye_spectra_dir, "B95.npz"))["mua"]
spectrum_B95 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B95)

spectrum_B97 = load_iad_results(os.path.join(dye_spectra_dir, "B97.npz"))["mua"]
spectrum_B97 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B97)

spectrum_B98 = load_iad_results(os.path.join(dye_spectra_dir, "B98.npz"))["mua"]
spectrum_B98 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B98)

spectrum_BJ6 = load_iad_results(os.path.join(dye_spectra_dir, "BJ6.npz"))["mua"]
spectrum_BJ6 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BJ6)

spectrum_BJ7 = load_iad_results(os.path.join(dye_spectra_dir, "BJ7.npz"))["mua"]
spectrum_BJ7 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BJ7)

spectrum_BJ8 = load_iad_results(os.path.join(dye_spectra_dir, "BJ8.npz"))["mua"]
spectrum_BJ8 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BJ8)

spectrum_BJ9 = load_iad_results(os.path.join(dye_spectra_dir, "BJ9.npz"))["mua"]
spectrum_BJ9 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BJ9)

spectrum_BIR = load_iad_results(os.path.join(dye_spectra_dir, "BIR.npz"))["mua"]
spectrum_BIR = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BIR)

spectrum_BIR2 = load_iad_results(os.path.join(dye_spectra_dir, "BIR2.npz"))["mua"]
spectrum_BIR2 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BIR2)

spectrum_BIR3 = load_iad_results(os.path.join(dye_spectra_dir, "BIR3.npz"))["mua"]
spectrum_BIR3 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BIR3)

spectrum_BI6 = load_iad_results(os.path.join(dye_spectra_dir, "BI6.npz"))["mua"]
spectrum_BI6 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BI6)

spectrum_BI7 = load_iad_results(os.path.join(dye_spectra_dir, "BI7.npz"))["mua"]
spectrum_BI7 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BI7)

spectrum_BI8 = load_iad_results(os.path.join(dye_spectra_dir, "BI8.npz"))["mua"]
spectrum_BI8 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BI8)

spectrum_BI9 = load_iad_results(os.path.join(dye_spectra_dir, "BI9.npz"))["mua"]
spectrum_BI9 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BI9)

spectrum_BI10 = load_iad_results(os.path.join(dye_spectra_dir, "BI10.npz"))["mua"]
spectrum_BI10 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BI10)

spectrum_BR1 = load_iad_results(os.path.join(dye_spectra_dir, "BR1.npz"))["mua"]
spectrum_BR1 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BR1)

spectrum_BR2 = load_iad_results(os.path.join(dye_spectra_dir, "BR2.npz"))["mua"]
spectrum_BR2 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BR2)

spectrum_BR3 = load_iad_results(os.path.join(dye_spectra_dir, "BR3.npz"))["mua"]
spectrum_BR3 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BR3)

spectrum_BR4 = load_iad_results(os.path.join(dye_spectra_dir, "BR4.npz"))["mua"]
spectrum_BR4 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BR4)

spectrum_BS0 = load_iad_results(os.path.join(dye_spectra_dir, "BS0.npz"))["mua"]
spectrum_BS0 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BS0)

spectrum_BF1 = load_iad_results(os.path.join(dye_spectra_dir, "BF1.npz"))["mua"]
spectrum_BF1 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF1)

spectrum_BF2 = load_iad_results(os.path.join(dye_spectra_dir, "BF2.npz"))["mua"]
spectrum_BF2 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF2)

spectrum_BF3 = load_iad_results(os.path.join(dye_spectra_dir, "BF3.npz"))["mua"]
spectrum_BF3 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF3)

spectrum_BF4 = load_iad_results(os.path.join(dye_spectra_dir, "BF4.npz"))["mua"]
spectrum_BF4 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF4)

spectrum_BF5 = load_iad_results(os.path.join(dye_spectra_dir, "BF5.npz"))["mua"]
spectrum_BF5 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF5)

spectrum_BF6 = load_iad_results(os.path.join(dye_spectra_dir, "BF6.npz"))["mua"]
spectrum_BF6 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF6)

spectrum_BF7 = load_iad_results(os.path.join(dye_spectra_dir, "BF7.npz"))["mua"]
spectrum_BF7 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF7)

spectrum_BF8 = load_iad_results(os.path.join(dye_spectra_dir, "BF8.npz"))["mua"]
spectrum_BF8 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF8)

spectrum_BF9 = load_iad_results(os.path.join(dye_spectra_dir, "BF9.npz"))["mua"]
spectrum_BF9 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF9)

spectrum_BF10A = load_iad_results(os.path.join(dye_spectra_dir, "BF10A.npz"))["mua"]
spectrum_BF10A = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF10A)

spectrum_BF10B = load_iad_results(os.path.join(dye_spectra_dir, "BF10B.npz"))["mua"]
spectrum_BF10B = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF10B)

spectrum_BF10C = load_iad_results(os.path.join(dye_spectra_dir, "BF10C.npz"))["mua"]
spectrum_BF10C = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_BF10C)

# for fi, fr in enumerate([spectrum_BF1, spectrum_BF2, spectrum_BF3]):
#     print(f"Forearm {fi}: {np.mean(fr):.3f} cm⁻¹")

reprod_dict = {
    "BI6": (spectrum_BI6, 55.4462 + 4.6275),
    "BI7": (spectrum_BI7, 59.6827),
    "BI8": (spectrum_BI8, 58.797),
    "BI9": (spectrum_BI9, 58.8317),
    "BI10": (spectrum_BI10, 59.2203),
}

# added_mat = list()
# for k, v in reprod_dict.items():
#     ratio = np.mean(spectrum_BIR/v[0])
#     add = v[1]/ratio - v[1]
#     added_mat.append(add)
#     print(f"{k}: {ratio:.2f} ==> {add:.2f}")
#
# print(f"Total added mass: {np.sum(added_mat):.2f}")

oxy = 0.2

oxy_phantoms = {
    0.7: spectrum_B67,
    0.8: spectrum_B66,
    0.9: spectrum_B90,
}

oxy_dict = {
    0.6: spectrum_BR4,
    0.7: spectrum_BR3,
    0.8: spectrum_BR2,
    0.9: spectrum_BR1,
}

deoxy_dict = {
    0.6: spectrum_BJ6,
    0.7: spectrum_BJ7,
    0.8: spectrum_BJ8,
    0.9: spectrum_BJ9,
}
plt.figure(figsize=(7, 5))
# plt.subplot(1, 2, 1)
# plt.plot(unmixing_wavelengths, oxy*hbo2_spectrum + (1-oxy)*hb_spectrum, label=f"real blood {100*oxy}%", color="red")
# plt.plot(unmixing_wavelengths, oxy_phantoms[oxy], label=f"{100*oxy:.0f}:{100*(1-oxy):.0f} mix measured")
# plt.plot(unmixing_wavelengths, oxy_dict[oxy] + deoxy_dict[oxy], label=f"phantom mix {100*oxy}%", color=DyeColors["B30"], linestyle=":")
# plt.plot(unmixing_wavelengths, hbo2_spectrum, label="HbO2", color="red")
# plt.plot(unmixing_wavelengths, spectrum_BJ6, label="IR-1061 (60%)", color=DyeColors["B30"], alpha=0.3)
# plt.plot(unmixing_wavelengths, spectrum_BJ7, label="IR-1061 (70%)", color=DyeColors["B30"], alpha=0.5)
# plt.plot(unmixing_wavelengths, spectrum_BJ8, label="IR-1061 (80%)", color=DyeColors["B30"], alpha=0.7)
# plt.plot(unmixing_wavelengths, spectrum_BJ9, label="IR-1061 (90%)", color=DyeColors["B30"], alpha=0.9)
# plt.plot(unmixing_wavelengths, spectrum_BIR, label="100% oxy: IR-1061", color=DyeColors["B30"])
# plt.plot(unmixing_wavelengths, spectrum_BIR2, label="IR-1061 (100%) 2", color=DyeColors["B30"], linestyle="--")
plt.plot(unmixing_wavelengths, spectrum_BIR3, label="HbO2 dye (IR-1061)", color=DyeColors["B30"])
# plt.plot(unmixing_wavelengths, 4.868*spectrum_B30, label="HbO2 dye (IR-1061)", color=DyeColors["B30"], linestyle="--")
# plt.plot(unmixing_wavelengths, spectrum_B98, label="Mixture 98:2", color=DyeColors["B30"], alpha=0.9)
# plt.plot(unmixing_wavelengths, spectrum_B97, label="67.4% oxy: Mix 97:3", color=DyeColors["B30"], alpha=0.9, linestyle="--")
# plt.plot(unmixing_wavelengths, spectrum_B95, label="52.4% oxy: Mix 95:5", color=DyeColors["B30"], alpha=0.75, linestyle="--")
# plt.plot(unmixing_wavelengths, spectrum_B93, label="30.7% oxy: Mix 93:7", color=DyeColors["B30"], alpha=0.6, linestyle="--")
# plt.plot(unmixing_wavelengths, spectrum_B92, label="Mixture 92:8", color=DyeColors["B30"], alpha=0.5)
# plt.plot(unmixing_wavelengths, spectrum_B90, label="0% oxy: Mix 90:10", color=DyeColors["B30"], alpha=0.5)
# plt.plot(unmixing_wavelengths, spectrum_BI6, label="IR-1061 (BI6)")
# plt.plot(unmixing_wavelengths, spectrum_BI7, label="IR-1061 (BI7)")
# plt.plot(unmixing_wavelengths, spectrum_BI8, label="IR-1061 (BI8)")
# plt.plot(unmixing_wavelengths, spectrum_BI9, label="IR-1061 (BI9)")
# plt.plot(unmixing_wavelengths, spectrum_BI10, label="IR-1061 (BI10)")

# plt.plot(unmixing_wavelengths, 0.2*hb_spectrum + 0.8*hbo2_spectrum, color="#ff0066", linestyle="--")
# plt.plot(unmixing_wavelengths, 0.4*hb_spectrum + 0.6*hbo2_spectrum, color="#ff00bc", linestyle="--")
# plt.plot(unmixing_wavelengths, 0.6*hb_spectrum + 0.4*hbo2_spectrum, color="#bc00ff", linestyle="--")
# plt.plot(unmixing_wavelengths, 0.8*hb_spectrum + 0.2*hbo2_spectrum, color="#6600ff", linestyle="--")
#
# plt.plot(unmixing_wavelengths, hb_spectrum, label="Hb", color="blue")
# plt.plot(unmixing_wavelengths, 3.185*spectrum_B42 + 0.826*spectrum_B43, label="Hb-Dye (3.185*IR-1043 + 0.826*Spectrasense)", linestyle="--", color="teal")
plt.plot(unmixing_wavelengths, spectrum_BS0, label="Hb-dye (Spectrasense-765)", color="teal")
# plt.plot(unmixing_wavelengths, spectrum_BR4, label="Spectrasense (40%)", color="teal", alpha=2*0.4)
# plt.plot(unmixing_wavelengths, spectrum_BR3, label="Spectrasense (30%)", color="teal", alpha=2*0.3)
# plt.plot(unmixing_wavelengths, spectrum_BR2, label="Spectrasense (20%)", color="teal", alpha=2*0.2)
# plt.plot(unmixing_wavelengths, spectrum_BR1, label="Spectrasense (10%)", color="teal", alpha=2*0.1)

# plt.plot(unmixing_wavelengths, hb_spectrum, label="Hb", color="blue")
# plt.plot(unmixing_wavelengths, spectrum_B50, label="Spectrasense (reproduction)", color="teal", linestyle="--")
# plt.plot(unmixing_wavelengths, scatter_B61, label="Spectrasense (Hb dye)", color="teal")

# plt.plot(unmixing_wavelengths, spectrum_BF1, label="Forearm 1: bvf: 2.5%, oxy: 0% (B90)", color="blue", alpha=0.75)
# plt.plot(unmixing_wavelengths, spectrum_BF2, label="Forearm 2: bvf: 2.5%, oxy: 50% (B95)", color="#ff00ff", alpha=0.75)
# plt.plot(unmixing_wavelengths, spectrum_BF3, label="Forearm 3: bvf: 2.5%, oxy: 100% (BIR)", color="red", alpha=0.75)

# plt.plot(unmixing_wavelengths, spectrum_BF4, label="Forearm 4: bvf: 4%, oxy: 100% (BIR)", color="red", alpha=1)
# plt.plot(unmixing_wavelengths, spectrum_BF5, label="Forearm 5: bvf: 4%, oxy: 50% (B95)", color="#ff00ff", alpha=1)
# plt.plot(unmixing_wavelengths, spectrum_BF6, label="Forearm 6: bvf: 4%, oxy: 0% (B90)", color="blue", alpha=1)
#
# plt.plot(unmixing_wavelengths, spectrum_BF7, label="Forearm 7: bvf: 1%, oxy: 100% (BIR)", color="red", alpha=0.5)
# plt.plot(unmixing_wavelengths, spectrum_BF8, label="Forearm 8: bvf: 1%, oxy: 50% (B95)", color="#ff00ff", alpha=0.5)
# plt.plot(unmixing_wavelengths, spectrum_BF9, label="Forearm 9: bvf: 1%, oxy: 0% (B90)", color="blue", alpha=0.5)

# plt.plot(unmixing_wavelengths, spectrum_BF10A, label="Forearm 10A: bvf: 0.5%, oxy: 100% (BIR)", color="red", alpha=0.5)
# plt.plot(unmixing_wavelengths, spectrum_BF10B, label="Forearm 10B: bvf: 5%, oxy: 0% (B95)", color="blue", alpha=1)
# plt.plot(unmixing_wavelengths, spectrum_BF10C, label="Forearm 10C: bvf: 3%, oxy: 70% (B90)", color="#ff007a", alpha=0.9)

plt.plot(unmixing_wavelengths, 0.03*spectrum_BIR3 + 0.97*spectrum_BS0, color="#34615C", linestyle="--")
plt.plot(unmixing_wavelengths, 0.08*spectrum_BIR3 + 0.92*spectrum_BS0, color="#53665b", linestyle="--")
plt.plot(unmixing_wavelengths, 0.15*spectrum_BIR3 + 0.85*spectrum_BS0, color="#735a4d", linestyle="--")
plt.plot(unmixing_wavelengths, 0.4*spectrum_BIR3 + 0.6*spectrum_BS0, color="#3a1f16", linestyle="--")

plt.ylim([1.147, 10.013])
plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
plt.legend()
# plt.subplot(1, 2, 1)
# plt.plot(unmixing_wavelengths, 5*0.156*spectrum_B50 + 0.8*spectrum_B56, label="16:84 mix Sp(c=1):Nigrosin (separately)", color="#2c3e50", linestyle="--")
# # plt.plot(unmixing_wavelengths, spectrum_B50, label="Spectrasense", color="teal")
# # plt.plot(unmixing_wavelengths, spectrum_B56, label="Nigrosin", color="black")
# plt.subplot(1, 2, 2)
# plt.plot(unmixing_wavelengths, scatter_B61, label="20:80 mix Sp(c=5):Nigrosin", color="#2c3e50")
# # plt.plot(unmixing_wavelengths, scatter_B50, label="Spectrasense-765(25.6 mg)", color="teal")
# plt.plot(unmixing_wavelengths, 5*0.156*scatter_B50 + 0.8*scatter_B56, label="16:84 mix Sp(c=1):Nigrosin (separately)", color="#2c3e50", linestyle="--")
# # plt.ylim([1, 10])
# plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
# plt.xlabel("Wavelength [nm]")
# plt.legend()


plt.tight_layout()
plt.savefig(f"/home/kris/Data/Dye_project/Plots/spectrum_comparison.png", dpi=400)
plt.close()


