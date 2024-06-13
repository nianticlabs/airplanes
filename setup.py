import setuptools

__version__ = "0.1.0"


setuptools.setup(
    name="airplanes",
    version=__version__,
    description="AirPlanes: Accurate Plane Estimation via 3D-Consistent Embeddings",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    project_urls={"Source": " https://github.com/nianticlabs/airplanes"},
)
