from setuptools import setup
import os

try:
    dependencies_managed_by_conda = os.environ['DEPENDENCIES_MANAGED_BY_CONDA'] == '1'
except KeyError:
    dependencies_managed_by_conda = False

docs_packages = [
    'mkdocstrings-python',
    'mkdocs-material',
]

setup(
    name='poseviz',
    version='0.1.7',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['poseviz', 'poseviz.components'],
    scripts=[],
    license='LICENSE',
    description='3D human pose visualizer with multi-person, multi-view support, built on Mayavi',
    long_description='',
    python_requires='>=3.6',
    install_requires=[] if dependencies_managed_by_conda else [
        'attrdict',
        'more-itertools',
        'transforms3d',
        'opencv-python',
        'numpy',
        'mayavi',
        'imageio',
        'cameralib @ git+https://github.com/isarandi/cameralib.git',
        'boxlib @ git+https://github.com/isarandi/boxlib.git',
    ],
    extras_require={
        'dev': docs_packages,
        'docs': docs_packages,
    },
)
