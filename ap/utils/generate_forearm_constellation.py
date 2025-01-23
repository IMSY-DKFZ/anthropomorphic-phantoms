import numpy as np
np.random.seed(42)

nr_forearms = 10
nr_vessels_per_forearm = 14
nr_vessels_per_oxy_level = 7

vessel_diameters = [3, 4, 5, 6]
oxy_levels = ["0%", "30%", "50%", "70%", "100%"]

forearm_dict = {f"Forearm {idx + 1}": list() for idx in range(nr_forearms)}


for oxy_level in oxy_levels:
    for diameter in vessel_diameters:
        possible_forearms = list(forearm_dict.keys())
        chosen_forearms = np.random.choice(possible_forearms, nr_vessels_per_oxy_level, replace=False)
        for possible_forearm in possible_forearms:
            if len(forearm_dict[possible_forearm]) > 14:
                possible_forearms.remove(possible_forearm)
        for forearm in chosen_forearms:
            forearm_dict[forearm].append((oxy_level, diameter))

for i, (k, v) in enumerate(forearm_dict.items()):
    print(f"\n{k}\n")
    for v_idx, ll in enumerate(v):
        print(f"{v_idx + 1}: Oxy: {ll[0]}, diameter: {ll[1]} mm")

# c_orig = 0.93
# c_target = 0.92
# m_new = 5.299
#
# m_add = c_orig*m_new / c_target - m_new
# print(m_add)
#
#
#
