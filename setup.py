# pylint: disable=missing-module-docstring
import setuptools

with open("README.md", 'r') as readme:
    long_description = readme.read()

with open("requirements.txt", 'r') as dependencies:
    requirements = [pkg.strip() for pkg in dependencies]

VERSION = "0.1.0"

setuptools.setup(
    name="HorizonGRound",
    version=VERSION,
    author="Mike S Wang",
    author_email="mike.wang@port.ac.uk",
    license="GPLv3",
    description=(
        "Forward-modelling of relativistic effects "
        "from tracer luminosity function.",
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MikeSWang/HorizonGRound/tree/v{}".format(VERSION),
    packages=['horizonground', 'horizonground.tests'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    setup_requires=['setuptools>=18.0', "cython>=0.19"],
    python_requires='>=3.6',
    project_urls={
        "Documentation": "https://mikeswang.github.io/HorizonGRound/",
        "Source": "https://github.com/MikeSWang/HorizonGRound/tree/v{}"\
            .format(VERSION),
    },
)
