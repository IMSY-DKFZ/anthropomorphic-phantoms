import os
import glob
import json
import latextable
from texttable import Texttable

try:
    run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
    print("This runner script is invoked in a bash script!")
except KeyError:
    run_by_bash: bool = False

if run_by_bash:
    base_path = os.environ["BASE_PATH"]
else:
    # In case the script is run from an IDE, the base path has to be set manually
    base_path = ""

pat_path = os.path.join(base_path, "Paper_Results", "PAT_Measurement_Correlation")
pat_results = [os.path.join(pat_path, f"PAT_spectrum_correlation_oxy_{oxy}_p0.json") for oxy in [0, 30, 50, 70, 100]]
pat_row = [f"{json.load(open(res_file))['r_value']:.2f}" for res_file in pat_results]

hsi_path = os.path.join(base_path, "Paper_Results", "HSI_Measurement_Correlation")
hsi_results = [os.path.join(hsi_path, f"HSI_spectrum_correlation_oxy_{oxy}.json") for oxy in [0, 30, 50, 70, 100]]
hsi_row = [f"{json.load(open(res_file))['r_value']:.2f}" for res_file in hsi_results]

table = Texttable()
table.set_cols_dtype(["t", "t", "t", "t", "t", "t"])
table.set_cols_align(["l", "c", "c", "c", "c", "c"])
table.set_cols_valign(["m", "m", "m", "m", "m", "m"])
table.add_row(["", r"0\% \ac*{oxy}", r"30\% \ac*{oxy}", r"50\% \ac*{oxy}", r"70\% \ac*{oxy}", r"100\% \ac*{oxy}"])
table.add_row(["PAT"] + pat_row)
table.add_row(["HSI"] + hsi_row)


outfile = os.path.join(base_path, "Paper_Results", "signal_correlation_results_table.txt")
with open(outfile, "w") as f:
    f.write(table.draw())

    f.writelines(["\n", "\n", "The following is the code for the latex table:", "\n", "\n"])

    f.write(latextable.draw_latex(table))
