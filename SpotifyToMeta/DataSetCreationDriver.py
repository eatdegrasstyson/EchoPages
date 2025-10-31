import subprocess
from pathlib import Path

#Driver script for generating training data from scratch!
#NOTE: All datasets must be installed and stored in the respective folders (MuVi, Music-Mouv\music-mouv, Trompa-Mer)
project_root = Path(__file__).resolve().parent.parent
scripts_dir = project_root / "SpotifyToMeta"

scripts_to_run = [
    "Music-Mouv_Processor.py",
    "Music-Mouv_Testing.py",
    "MuVi_Processor.py",
    "Tompa-Mer_Processor.py",
    "Tompa-Mer_Spotify.py",
    "Tompa-Mer_Final.py",
    "Tompa-Mer_Final.py",
    "CombineDatasets.py",
    "Computeetadata.py"
]

for script_name in scripts_to_run:
    script_path = scripts_dir / script_name
    print(f"Running {script_path}...")

    #Run script and wait for it to finish
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    
    #Output script prints
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    #Stop if script failed
    if result.returncode != 0:
        print(f"Script {script_name} failed with exit code {result.returncode}")
        break

print("All scripts executed!")
