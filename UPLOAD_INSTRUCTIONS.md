# Upload FarmGrow AI to GitHub

1. Unzip `FarmGrow-AI_GitHub_Ready.zip` on your computer.
2. On GitHub, create a new **public** repository named `FarmGrow-AI`.
   Leave the automatic README, .gitignore, and license options unchecked.
3. In the new repository, choose **Add file > Upload files**.
4. Open the extracted `FarmGrow-AI` folder and select/drag its **contents**
   (including `README.md`, `.gitignore`, the Python files, CSV, and `.spec`).
   Do not upload the enclosing folder as an extra directory layer.
5. Commit the upload. Verify that `main.py`, `requirements.txt`,
   `plant_species.csv`, and `README.md` appear at the repository root.
6. Inspect the README in GitHub and confirm the application runs in a compatible
   local Python environment.

This package is prepared from your original CPSC-4370 Project V4.zip.
Original Python source files, requirements, plant data, and PyInstaller spec were
copied without modification. `README.md` and `.gitignore` were added.
The prior `.venv`, `.mypy_cache`, and `build` folders were deliberately omitted.

Do not upload the original 65 MB ZIP directly as a repository file.
