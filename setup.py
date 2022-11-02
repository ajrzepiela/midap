from setuptools import setup, find_packages

setup(
    name="midap",
    version="0.1.2",
    description='A package for cell segmentation and tracking.',
    author='Oschmann  Franziska, Fluri Janis',
    author_email='franziska.oschmann@id.ethz.ch, janis.fluri@id.ethz.ch',
    python_requires='>=3.8, <4',
    keywords='Segmentation, Tracking, Biology',
    packages=find_packages(include=["midap.*"]),
    project_urls={'Midap': 'https://gitlab.ethz.ch/oschmanf/ackermann-bacteria-segmentation/'},
)
