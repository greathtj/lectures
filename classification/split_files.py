import os
import random
import shutil

def split_files_by_ratio(source_folder, dest_folder_a, dest_folder_b, dest_folder_c, ratio=(7, 2, 1)):
    """
    Randomly splits files from a source folder into three destination folders
    based on a given ratio.

    Args:
        source_folder (str): Path to the folder containing the files to be split.
        dest_folder_a (str): Path to the first destination folder (e.g., for 70%).
        dest_folder_b (str): Path to the second destination folder (e.g., for 20%).
        dest_folder_c (str): Path to the third destination folder (e.g., for 10%).
        ratio (tuple): A tuple of three integers representing the desired ratio
                       (e.g., (7, 2, 1) for 70%, 20%, 10%).
    """

    # Validate ratio
    if not (isinstance(ratio, tuple) and len(ratio) == 3 and all(isinstance(r, int) and r >= 0 for r in ratio) and sum(ratio) > 0):
        raise ValueError("Ratio must be a tuple of three non-negative integers that sum to greater than zero.")

    # Create destination folders if they don't exist
    for folder in [dest_folder_a, dest_folder_b, dest_folder_c]:
        os.makedirs(folder, exist_ok=True)

    # Get all files from the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    random.shuffle(files) # Shuffle the files randomly

    total_files = len(files)
    if total_files == 0:
        print(f"No files found in '{source_folder}'.")
        return

    # Calculate split points
    total_ratio_sum = sum(ratio)
    split_point_a = int(total_files * (ratio[0] / total_ratio_sum))
    split_point_b = int(total_files * ((ratio[0] + ratio[1]) / total_ratio_sum))

    # Split the files based on calculated points
    files_a = files[:split_point_a]
    files_b = files[split_point_a:split_point_b]
    files_c = files[split_point_b:]

    # Copy files to their respective destination folders
    print(f"Splitting {total_files} files from '{source_folder}'...")

    for f in files_a:
        shutil.copy(os.path.join(source_folder, f), os.path.join(dest_folder_a, f))
    print(f"Copied {len(files_a)} files to '{dest_folder_a}' (approx. {ratio[0]/total_ratio_sum:.1%})")

    for f in files_b:
        shutil.copy(os.path.join(source_folder, f), os.path.join(dest_folder_b, f))
    print(f"Copied {len(files_b)} files to '{dest_folder_b}' (approx. {ratio[1]/total_ratio_sum:.1%})")

    for f in files_c:
        shutil.copy(os.path.join(source_folder, f), os.path.join(dest_folder_c, f))
    print(f"Copied {len(files_c)} files to '{dest_folder_c}' (approx. {ratio[2]/total_ratio_sum:.1%})")

    # Verify the total count (due to integer truncation, may not be exact)
    print(f"Total files copied: {len(files_a) + len(files_b) + len(files_c)}. Original files: {total_files}")

# --- Example Usage ---
if __name__ == "__main__":
    this_class = "full"
    # Define your source and destination folders
    SOURCE_DIR = f"C:/Users/greathtj/Desktop/photos/{this_class}"
    DEST_TRAIN = f"C:/Users/greathtj/Desktop/dataset/train/{this_class}"
    DEST_VALID = f"C:/Users/greathtj/Desktop/dataset/val/{this_class}"
    DEST_TEST = f"C:/Users/greathtj/Desktop/dataset/test/{this_class}"

    # Call the function to split the files with the default 7:2:1 ratio
    try:
        split_files_by_ratio(SOURCE_DIR, DEST_TRAIN, DEST_VALID, DEST_TEST)
        print("\nFile splitting complete!")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
