from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup

# Packaging
# ========================================
# Clean: python setup.py clean
# Build: python -mbuild . --sdist --wheel
# Test:  pytest --pyargs rpc3.tests


def process_readme() -> str | None:
    """Copy the README file into /src and return its content."""
    readme_file = Path(__file__).parent / "README.rst"
    shutil.copyfile(readme_file, readme_file.parent / "src" / readme_file.name)
    with readme_file.open(encoding="utf-8") as f:
        return f.read()
    return None


setup(
    name="rpc3-file",
    version="1.0.0rc3",
    license="BSD-2-Clause License",
    description="Read/write access to data files in RPC3 file format.",
    long_description=process_readme(),  # Use README.rst as long description
    long_description_content_type="text/x-rst",  # Specify format
    url="http://github.com/a-ma72/rpc3-file",
    author="Andreas Martin",
    setup_requires=["wheel"],
    python_requires=">=3.7",
    install_requires=["numpy>=1.19", "tqdm"],
    package_dir={"rpc3": "src", "rpc3.img": "img", "rpc3.tests": "tests"},
    include_package_data=True,
    package_data={"rpc3": ["README.rst"], "rpc3.img": ["*.png"]},
    classifiers=[
        "Development Status :: 6 - Mature",
        "Environment :: Console",
        "Framework :: Buildout :: Extension",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD-2-Clause License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
