from setuptools import setup

setup(
    name='poseviz',
    version='0.1.0',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['poseviz'],
    scripts=[],
    license='LICENSE',
    description='3D human pose visualizer with multi-person, multi-view support, built on Mayavi',
    long_description='',
    install_requires=[
        'attrdict', 'more-itertools', 'transforms3d', 'opencv-python', 'numpy'
        # and mayavi, but that is best installed via conda
    ],
)
