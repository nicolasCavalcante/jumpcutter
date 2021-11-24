from pathlib import Path

from setuptools import find_packages, setup

SELF_PATH = Path(__file__).parent.absolute()


def read(path: Path):
    with open(path, "r") as f:
        return f.read()


setup(
    name="jumpcutter",
    description="Automatically edits vidx. Explanation here: https://www.youtube.com/watch?v=DQ8orIurGxw",
    entry_points={
        "console_scripts": ["jumpcutter=jumpcutter.cli:app"],
    },
    author="Nicolas Garcia Cavalcante",
    author_email="nicolasgcavalcante@gmail.com",
    packages=find_packages(include=["jumpcutter", "jumpcutter.*"]),
    include_package_data=True,
    license="MIT license",
    keywords="jumpcutter",
    url="https://github.com/nicolasCavalcante/jumpcutter",
    long_description=read(SELF_PATH / "README.md"),
)
