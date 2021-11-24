import platform
import subprocess
from pathlib import Path

SEP = "&" if platform.system() == "Windows" else ";"
SELF_PATH = Path(__file__).parent.absolute()
NBS_PATH = SELF_PATH / "notebooks"
DOIT_CONFIG = {"default_tasks": ["format", "formatnb", "pytest"]}


def syscmd(string):
    subprocess.call(string, shell=True)
    return True


def task_format():
    """makes code organized and pretty"""
    nparts = len(SELF_PATH.parts)
    for filepath in SELF_PATH.glob("**/*.py"):
        yield {
            "name": "/".join(filepath.parts[nparts:]),
            "actions": [
                (
                    "autoflake -i -r --expand-star-imports"
                    + " --remove-all-unused-imports"
                    + " --remove-duplicate-keys --remove-unused-variables %s"
                    + " %s isort %s %s black --line-length 79 %s"
                )
                % (filepath, SEP, filepath, SEP, filepath)
            ],
            "file_dep": [filepath],
            "verbosity": 2,
        }


def task_formatnb():
    """makes notebooks organized and pretty"""
    nparts = len(NBS_PATH.parts)
    for filepath in NBS_PATH.glob("*.ipynb"):
        filename = filepath.as_posix()
        yield {
            "name": "/".join(filepath.parts[nparts:]),
            "actions": [
                ('nbqa isort "%s" %s nbqa black "%s"')
                % (filename, SEP, filename)
            ],
            "file_dep": [filepath],
            "verbosity": 2,
        }


def task_pytest():
    """run pytests under tests folder"""
    return {"actions": [lambda: syscmd("pytest tests/")], "verbosity": 2}


def instalation_config(action_str):
    return {
        "actions": [lambda: syscmd(action_str)],
        "verbosity": 2,
    }


def task_devinstall():
    """install development packages"""
    return instalation_config('pip install -e "' + str(SELF_PATH) + '"')


def task_install():
    """install package"""
    return instalation_config('pip install "' + str(SELF_PATH) + '"')


def task_uninstall():
    """uninstall package"""
    return {
        "actions": [lambda: syscmd("pip uninstall jumpcutter -y")],
        "verbosity": 2,
    }
