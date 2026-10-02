"""Create and remove the optional BirdCrop Windows desktop shortcut."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

SHORTCUT_NAME = "BirdCrop.lnk"
BATCH_FILENAME = "birdcrop-dragdrop.bat"
# User may tweak the birdcrop arguments after -m birdcrop --shortcut below.
# This can be edited in the shortcut properties after creation, but the default is set here.
PYTHON_ARGUMENTS = "-m birdcrop --shortcut --confidence 0.33 --"


def _windows_components():
    """Load pywin32 only when the Windows-only command is actually used."""
    try:
        win32_client = importlib.import_module("win32com.client")
        shell = importlib.import_module("win32com.shell.shell")
        shellcon = importlib.import_module("win32com.shell.shellcon")
    except ImportError as exc:
        raise RuntimeError(
            "The Windows shortcut support requires the pywin32 package. "
            "Reinstall bird-crop or run: python -m pip install pywin32"
        ) from exc
    return win32_client, shell, shellcon


def _paths() -> tuple[Path, Path]:
    _, shell, shellcon = _windows_components()
    batch_path = Path(__file__).with_name(BATCH_FILENAME).resolve()
    desktop_path = shell.SHGetFolderPath(0, shellcon.CSIDL_DESKTOPDIRECTORY, None, 0)
    return batch_path, Path(desktop_path) / SHORTCUT_NAME


def _same_path(first: str | Path, second: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(str(first))) == os.path.normcase(
        os.path.abspath(str(second))
    )


def _interpreter_path() -> Path:
    """Find the interpreter belonging to the environment containing this package."""
    package_dir = Path(__file__).resolve().parent
    if len(package_dir.parents) >= 3:
        environment_root = package_dir.parents[2]
        environment_interpreter = environment_root / "Scripts" / "python.exe"
        if environment_interpreter.is_file():
            return environment_interpreter
    return Path(sys.executable).resolve()


def _shortcut_object(shortcut_path: Path):
    win32_client, _, _ = _windows_components()
    shell = win32_client.Dispatch("WScript.Shell")
    return shell.CreateShortcut(str(shortcut_path))


def create_shortcut() -> int:
    if sys.platform != "win32":
        print("The desktop shortcut is supported on Windows only.")
        return 0

    try:
        batch_path, shortcut_path = _paths()
        if not batch_path.is_file():
            print(f"Installed batch file not found: {batch_path}", file=sys.stderr)
            return 1

        # Das eigentliche Ziel ist die systemeigene cmd.exe (das vermeidet das Shortcut->Symlink-Problem)
        expected_target = os.environ.get("COMSPEC", "cmd.exe")
        python_path = _interpreter_path()
        shortcut_arguments = f'/c "{python_path}" {PYTHON_ARGUMENTS}'
        if shortcut_path.is_file():
            existing = _shortcut_object(shortcut_path)
            is_legacy_shortcut = _same_path(existing.TargetPath, batch_path) and not existing.Arguments
            is_current_shortcut = _same_path(existing.TargetPath, expected_target) and existing.Arguments == shortcut_arguments
            if existing.TargetPath and not (is_legacy_shortcut or is_current_shortcut):
                print(
                    f"A different shortcut already exists at {shortcut_path}.",
                    file=sys.stderr,
                )
                return 1

        shortcut = _shortcut_object(shortcut_path)
        shortcut.TargetPath = str(expected_target)
        shortcut.Arguments = shortcut_arguments
        shortcut.WorkingDirectory = str(batch_path.parent)
        shortcut.Description = "BirdCrop: process images by drag and drop"
        # Optik: Das Icon direkt aus der Python.exe extrahieren
        shortcut.IconLocation = str(python_path) + ",0" 
        shortcut.Save()
    except Exception as exc:
        print(f"Could not create the desktop shortcut: {exc}", file=sys.stderr)
        return 1

    print(f"Desktop shortcut created: {shortcut_path}")
    print("Images and folders can now be dropped onto it.")
    return 0


def remove_shortcut() -> int:
    if sys.platform != "win32":
        print("The desktop shortcut is supported on Windows only.")
        return 0

    try:
        batch_path, shortcut_path = _paths()
        if not shortcut_path.is_file():
            print(f"No BirdCrop shortcut found at {shortcut_path}.")
            return 0

        expected_target = os.environ.get("COMSPEC", "cmd.exe")
        python_path = _interpreter_path()
        shortcut_arguments = f'/c "{python_path}" {PYTHON_ARGUMENTS}'
        shortcut = _shortcut_object(shortcut_path)
        is_legacy_shortcut = _same_path(shortcut.TargetPath, batch_path) and not shortcut.Arguments
        is_current_shortcut = (
            _same_path(shortcut.TargetPath, expected_target)
            and shortcut.Arguments == shortcut_arguments
        )
        if not (is_legacy_shortcut or is_current_shortcut):
            print(
                f"The shortcut was modified and was left untouched: {shortcut_path}",
                file=sys.stderr,
            )
            return 1

        shortcut_path.unlink()
    except Exception as exc:
        print(f"Could not remove the desktop shortcut: {exc}", file=sys.stderr)
        return 1

    print(f"Desktop shortcut removed: {shortcut_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or remove the optional BirdCrop desktop shortcut on Windows."
    )
    parser.add_argument(
        "--remove-shortcut",
        action="store_true",
        help="Remove the unchanged BirdCrop shortcut from the desktop.",
    )
    args = parser.parse_args()
    return remove_shortcut() if args.remove_shortcut else create_shortcut()


if __name__ == "__main__":
    raise SystemExit(main())
