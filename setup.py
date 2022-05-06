from setuptools import find_packages, setup

requirements = [
    'bioformats>=0.1.15',
    'imageio>=2.19.0',
    'javabridge>=1.0.19',
    'keras>=2.8.0',
    'matplotlib>=3.5.1',
    'napari>=0.4.15',
    'numpy>=1.21.1',
    'pandas>=1.3.1',
    'Pillow>=9.1.0',
    'psutil>=5.9.0',
    'PySimpleGUI>=4.59.0',
    'PyYAML>=6.0',
    'scikit_learn>=1.0.2',
    'scipy==1.7.1',
    'skimage>=0.0',
    'tensorflow>=2.8.0',
    'tqdm>=4.62.0',
]

setup(
    name="midap_utils",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
