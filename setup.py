"""
setup_env.py

Creates a Python virtual environment and installs required packages
for the ECG PTB-XL AFIB Detection Pipeline.

Usage:
    python setup_env.py

After setup:
    Windows:
        venv\\Scripts\\activate
        python src\\main.py

    macOS/Linux:
        source venv/bin/activate
        python src/main.py
"""

import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def get_venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run_command(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(str(x) for x in command))

    subprocess.check_call(command)


def create_venv() -> None:
    if VENV_DIR.exists():
        print(f"\nVirtual environment already exists: {VENV_DIR}")
        return

    print(f"\nCreating virtual environment: {VENV_DIR}")

    run_command([
        sys.executable,
        "-m",
        "venv",
        str(VENV_DIR),
    ])


def install_requirements() -> None:
    venv_python = get_venv_python()

    if not venv_python.exists():
        raise FileNotFoundError(f"Venv Python not found: {venv_python}")

    print("\nUpgrading pip...")

    run_command([
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
    ])

    if REQUIREMENTS_FILE.exists():
        print("\nInstalling packages from requirements.txt...")

        run_command([
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_FILE),
        ])
    else:
        print("\nrequirements.txt not found. Skipping package installation.")


def print_next_steps() -> None:
    print("\nSetup completed.")

    if os.name == "nt":
        print("\nNext steps:")
        print(r"  venv\Scripts\activate")
        print(r"  python src\main.py")
    else:
        print("\nNext steps:")
        print("  source venv/bin/activate")
        print("  python src/main.py")


def main() -> None:
    print("=" * 70)
    print("ECG PTB-XL AFIB Detection Pipeline - Environment Setup")
    print("=" * 70)

    create_venv()
    install_requirements()
    print_next_steps()


if __name__ == "__main__":
    main()