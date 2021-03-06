# pylint: disable=missing-module-docstring
import setuptools

with open("README.md", 'r') as description:
    long_description = description.read()

with open("requirements.txt", 'r') as dependencies:
    requirements = [pkg.strip() for pkg in dependencies]

with open("version.txt", 'r') as version_info:
    version_tag, version = [v.strip() for v in version_info]
    if version_tag == 'latest':
        branch = 'master'
    else:
        branch = version_tag

setuptools.setup(
    name="HorizonGRound",
    version=version,
    author="Mike S Wang",
    author_email="mike.wang@port.ac.uk",
    license="GPLv3",
    description=(
        "Forward-modelling of relativistic effects "
        "from the tracer luminosity function."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MikeSWang/HorizonGRound/",
    packages=['horizonground', 'horizonground.tests'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
    project_urls={
        "Documentation": "https://mikeswang.github.io/HorizonGRound",
        "Source": "https://github.com/MikeSWang/HorizonGRound/",
    },
)
