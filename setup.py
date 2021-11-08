from setuptools import setup

setup(
    name='poseviz',
    version='0.1.1',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['poseviz', 'poseviz.components'],
    scripts=[],
    license='LICENSE',
    description='3D human pose visualizer with multi-person, multi-view support, built on Mayavi',
    long_description='',
    python_requires='>=3.6',
    # Required libraries are listed in meta.yaml for conda packaging
    # install_requires=[
    #   'attrdict', 'more-itertools', 'transforms3d', 'opencv-python', 'numpy', 'mayavi', 'imageio']
)
