from setuptools import setup, find_packages

setup(
    name="midap",
    version="0.3.11",
    description='A package for cell segmentation and tracking.',
    author='Oschmann  Franziska, Fluri Janis',
    author_email='franziska.oschmann@id.ethz.ch, janis.fluri@id.ethz.ch',
    python_requires='>=3.9, <4',
    keywords='Segmentation, Tracking, Biology',
    packages=find_packages(include=["midap.*"]),
    project_urls={'Midap': 'https://gitlab.ethz.ch/oschmanf/ackermann-bacteria-segmentation/'},
    entry_points={
        'console_scripts': [
            'midap = midap.main:run_module',
            'correct_segmentation = midap.apps.correct_segmentation:main',
            'midap_download = midap.apps.download_files:main',
        ],
    },
)
