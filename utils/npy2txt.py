import numpy as np
import os

# Load the .npy file
file_path = "/home/khanm/workfolder/InstantSplat/output_infer/sora/Santorini/2_views/pose/ours_1000/pose_interpolated.npy"  # Replace with your actual file path
data = np.load(file_path)

# Define the output text file path
txt_file_path = os.path.splitext(file_path)[0] + ".txt"

# Save with structured formatting
with open(txt_file_path, "w") as f:
    for i, slice_ in enumerate(data):
        f.write(f"Slice {i}:\n")
        np.savetxt(f, slice_, fmt="%.6f")
        f.write("\n")

print(f"Saved structured .npy contents to: {txt_file_path}")
