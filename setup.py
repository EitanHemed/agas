import setuptools

VERSION = "0.0.1"

INSTALL_REQUIRES = [
    'numpy>=1.13.3',
    'pandas>=0.25',
]

EXTRAS_REQUIRE = {}

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
]

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name="tspair",
    version=VERSION,
    author="Eitan Hemed",
    author_email="Eitan.Hemed@gmail.com",
    description="Time series pairing",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/EitanHemed/tspair",
    project_urls={
        "Bug Tracker": "https://github.com/EitanHemed/tspair/issues",
    },
    classifiers=CLASSIFIERS,
    package_dir={"": "tspair"},
    packages=setuptools.find_packages(where="tspair"),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifies=CLASSIFIERS,
)
