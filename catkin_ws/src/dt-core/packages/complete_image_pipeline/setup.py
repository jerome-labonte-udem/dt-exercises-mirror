## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        'complete_image_pipeline', 
        'complete_image_pipeline_tests',
        'duckietown_segmaps',
        'duckietown_segmaps_tests',
        'localization_templates',
        'localization_templates_tests',
    ],
    package_dir={'': 'include'},
)

setup(**setup_args)
