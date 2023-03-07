from setuptools import setup
import os

try:
    dependencies_managed_by_conda = os.environ['DEPENDENCIES_MANAGED_BY_CONDA'] == '1'
except KeyError:
    dependencies_managed_by_conda = False

setup(
    name='poseviz',
    version='0.1.4',
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
        'imageio']
)
