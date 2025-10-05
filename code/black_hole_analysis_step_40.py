import os
import numpy as np

output_directory = r"C:\Users\admin\Desktop\UWT 128 BH Analysis Data"
correction_factor = 4.91e-19  # Adjust based on validation

for filename in os.listdir(output_directory):
    if filename.startswith("black_hole_analysis_step_") and filename.endswith(".txt"):
        with open(os.path.join(output_directory, filename), 'r') as f:
            lines = f.readlines()
        corrected_lines = []
        for line in lines:
            if "Compact" in line:
                parts = line.split()
                if len(parts) > 4 and parts[4].replace('e', 'E').replace('+', '').replace('-', 'E-').lstrip('0'):
                    compact = float(parts[4])
                    corrected_compact = compact * correction_factor
                    line = line.replace(parts[4], f"{corrected_compact:.3e}")
            corrected_lines.append(line)
        with open(os.path.join(output_directory, f"corrected_{filename}"), 'w') as f:
            f.writelines(corrected_lines)
print("Compactness correction applied!")