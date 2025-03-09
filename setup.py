from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup

# Packaging
# ========================================
# Clean: python setup.py clean
# Build: python -mbuild . --sdist --wheel
# Test:  pytest --pyargs rpc3.tests

long_description = (
"""
About RPC III
=============

The RPC III file format, developed by MTS Systems Corporation, is widely used in vehicle durability
testing and simulation. This format is structured as sequential, fixed-length files, with each record
being 512 bytes. The file consists of a standard header followed by data records. The header typically
includes metadata such as the file creation date, channel information, and other configuration details
stored as keyword-value pairs.

The format supports up to 256 data channels and 1024 parameters, with specific limitations on property
names (maximum 32 characters) and values (maximum 96 characters). These files are often used in
conjunction with MTS's RPC Pro and RPC Connect software, which facilitate the analysis and simulation
of vehicle response to road conditions.

To work with RPC III files, tools like the MTS DataPlugin are available, allowing for the reading and
writing of these files within different software environments, such as National Instruments' LabVIEW.
The files typically carry extensions like .rsp or .tim.

If you're looking for a more detailed description of the format or specific documentation, it's usually
included in the software manuals provided by MTS or in the release notes of tools like the MTS


About this module
=================

This module provides functionality to read and write time history data from
RPC III format data files, commonly used in vehicle durability testing.
The module supports the extraction and modification of metadata, such as
channel information and test parameters, as well as the ability to handle
the structured data records stored within these files.

Key Features:
-------------
- Read and parse RPC III (.rsp, .tim) files.
- Write data and metadata back to RPC III files.
- Support for up to 256 data channels and 1024 parameters.
- Handles fixed-length records with standard headers.
"""
)


def process_readme() -> str | None:
    """Copy the README file into /src and return its content."""
    readme_file = Path(__file__).parent / "README.rst"
    shutil.copyfile(readme_file, readme_file.parent / "src" / readme_file.name)
    # with readme_file.open(encoding="utf-8") as f:
    #     content = f.read()
    return long_description


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
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
