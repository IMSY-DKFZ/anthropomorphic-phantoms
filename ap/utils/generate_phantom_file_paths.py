import re


def generate_file_paths(file_path):
    # Extract the number from the file path using regex
    matches = re.findall(r"_(\d+)_", file_path)

    if matches:
        nr = int(matches[-1])

        # Generate file paths for nr-1, nr, and nr+1
        file_path_minus_1 = file_path.replace(f"_{nr}_", f"_{nr - 1}_")
        file_path_nr = file_path
        file_path_plus_1 = file_path.replace(f"_{nr}_", f"_{nr + 1}_")

        # Return the list of file paths
        return [file_path_minus_1, file_path_nr, file_path_plus_1]
    else:
        raise ValueError("No number found in the file path")


if __name__ == "__main__":
    file_path = "_32_xxxx_123_x.hdf5"
    file_paths = generate_file_paths(file_path)

    print(file_paths)