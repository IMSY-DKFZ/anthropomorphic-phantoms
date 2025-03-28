import os
import json
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

pat_path = os.path.join(base_path, "Paper_Results", "Oxy_Results", "pat_results.json")
pat_results = json.load(open(pat_path))

hsi_path = os.path.join(base_path, "Paper_Results", "Oxy_Results", "hsi_results.json")
hsi_results = json.load(open(hsi_path))

lsu_row1 = (f"\multirow{{2}}{{5em}}{{LSU}} & {100*pat_results['Entire Phantom LSU Error Mean']:.1f}$\pm${100*pat_results['Entire Phantom LSU Error Std']:.1f} & "
           f"{100*pat_results['Vessels-only LSU Error Mean']:.1f}$\pm${100*pat_results['Vessels-only LSU Error Std']:.1f} & "
           f"{100*hsi_results['Entire Phantom LSU Error Mean']:.1f}$\pm${100*hsi_results['Entire Phantom LSU Error Std']:.1f} & "
           f"{100*hsi_results['Vessels-only LSU Error Mean']:.1f}$\pm${100*hsi_results['Vessels-only LSU Error Std']:.1f} \\\\")

lsu_row2 = (f"& [{100*pat_results['Entire Phantom LSU Error CI'][0]:.1f}, {100*pat_results['Entire Phantom LSU Error CI'][1]:.1f}] & "
            f"[{100*pat_results['Vessels-only LSU Error CI'][0]:.1f}, {100*pat_results['Vessels-only LSU Error CI'][1]:.1f}] & "
            f"[{100*hsi_results['Entire Phantom LSU Error CI'][0]:.1f}, {100*hsi_results['Entire Phantom LSU Error CI'][1]:.1f}] & "
            f"[{100*hsi_results['Vessels-only LSU Error CI'][0]:.1f}, {100*hsi_results['Vessels-only LSU Error CI'][1]:.1f}] \\\\")

cal_lsu_row1 = (f"\multirow{{2}}{{5em}}{{Calibrated LSU}} & {100*pat_results['Entire Phantom Cal. LSU Error Mean']:.1f}$\pm${100*pat_results['Entire Phantom Cal. LSU Error Std']:.1f} & "
                f"{100*pat_results['Vessels-only Cal. LSU Error Mean']:.1f}$\pm${100*pat_results['Vessels-only Cal. LSU Error Std']:.1f} & "
                f"{100*hsi_results['Entire Phantom Cal. LSU Error Mean']:.1f}$\pm${100*hsi_results['Entire Phantom Cal. LSU Error Std']:.1f} & "
                f"{100*hsi_results['Vessels-only Cal. LSU Error Mean']:.1f}$\pm${100*hsi_results['Vessels-only Cal. LSU Error Std']:.1f} \\\\")

cal_lsu_row2 = (f"& [{100*pat_results['Entire Phantom Cal. LSU Error CI'][0]:.1f}, {100*pat_results['Entire Phantom Cal. LSU Error CI'][1]:.1f}] & "
                f"[{100*pat_results['Vessels-only Cal. LSU Error CI'][0]:.1f}, {100*pat_results['Vessels-only Cal. LSU Error CI'][1]:.1f}] & "
                f"[{100*hsi_results['Entire Phantom Cal. LSU Error CI'][0]:.1f}, {100*hsi_results['Entire Phantom Cal. LSU Error CI'][1]:.1f}] & "
                f"[{100*hsi_results['Vessels-only Cal. LSU Error CI'][0]:.1f}, {100*hsi_results['Vessels-only Cal. LSU Error CI'][1]:.1f}] \\\\")

fluence_comp_row1 = (f"\multirow{{2}}{{5em}}{{Fluence compensation}} & {100*pat_results['Entire Phantom Fluence Comp. Error Mean']:.1f}$\pm${100*pat_results['Entire Phantom Fluence Comp. Error Std']:.1f} & "
                     f"{100*pat_results['Vessels-only Fluence Comp. Error Mean']:.1f}$\pm${100*pat_results['Vessels-only Fluence Comp. Error Std']:.1f} &"
                     f" & "
                     f"\\\\")

fluence_comp_row2 = (f"& [{100*pat_results['Entire Phantom Fluence Comp. Error CI'][0]:.1f}, {100*pat_results['Entire Phantom Fluence Comp. Error CI'][1]:.1f}] & "
                     f"[{100*pat_results['Vessels-only Fluence Comp. Error CI'][0]:.1f}, {100*pat_results['Vessels-only Fluence Comp. Error CI'][1]:.1f}] &"
                     f" & "
                     f"\\\\")

table = Texttable()
table.add_row(["", "PAT", "PAT", "HSI", "HSI"])
table.add_row(["", "Entire Phantom", "Vessels-only", "Entire Phantom", "Vessels-only"])

table.add_row(["LSU"] + [res.replace("$\pm$", "±").replace("\\\\", "") for res in lsu_row1.split("&")][1:])
table.add_row(["LSU"] + [res.replace("\\\\", "") for res in lsu_row2.split("&")][1:])
table.add_row(["Calibrated LSU"] + [res.replace("$\pm$", "±").replace("\\\\", "") for res in cal_lsu_row1.split("&")][1:])
table.add_row(["Calibrated LSU"] + [res.replace("\\\\", "") for res in cal_lsu_row2.split("&")][1:])
table.add_row(["Fluence compensation"] + [res.replace("$\pm$", "±").replace("\\\\", "") for res in fluence_comp_row1.split("&")][1:])
table.add_row(["Fluence compensation"] + [res.replace("\\\\", "") for res in fluence_comp_row2.split("&")][1:])

outfile = pat_path.replace("pat_results.json", "oxy_results_table.txt")
with open(outfile, "w") as f:
    f.write(table.draw())

    f.writelines(["\n", "\n", "The following is the code for the latex table:", "\n", "\n"])

    f.write("\t \t" + lsu_row1 + "\n")
    f.write("\t \t" + lsu_row2 + "\n")
    f.write("\t \t" + "\hline" + "\n")
    f.write("\t \t" + cal_lsu_row1 + "\n")
    f.write("\t \t" + cal_lsu_row2 + "\n")
    f.write("\t \t" + "\hline" + "\n")
    f.write("\t \t" + fluence_comp_row1.replace(" & & ",
                                                " & \multicolumn{2}{c}{\multirow{2}{*}{not applicable}}") + "\n")
    f.write("\t \t" + fluence_comp_row2.replace(" & & ", " & \multicolumn{2}{c}{} ") + "\n")
