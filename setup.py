from setuptools import setup, find_packages
import platform


def get_requirements():
    requirements=['btrack==0.4.6',
                        'coverage>=7.3.2',
                        'gitpython>=3.1.40',
                        'jupyterlab==4.0.8',
                        'matplotlib>=3.3.0,<=3.6.3',
                        'napari[all]',
                        'omnipose==0.4.4',
                        'opencv-python>=4.8.1',
                        'pandas>=2.0.2',
                        'psutil>=5.9.5',
                        'pytest>=7.4.3',
                        'scikit-image>=0.19.3,<=0.20.0',
                        'stardist>=0.8.5',
                        'tensorflow==2.13.0',
                        'tqdm>=4.65.0']

    if platform.processor() == 'arm':
        requirements+=['tensorflow-metal']

    return requirements
    

setup(
    name="midap",
    version="0.3.17",
    description='A package for cell segmentation and tracking.',
    author='Oschmann  Franziska, Fluri Janis',
    author_email='franziska.oschmann@id.ethz.ch, janis.fluri@id.ethz.ch',
    python_requires='>=3.9, <4',
    download_url = 'https://github.com/Microbial-Systems-Ecology/midap/archive/refs/tags/0.3.15.tar.gz',
    keywords='Segmentation, Tracking, Biology',
    install_requires=get_requirements(),
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
