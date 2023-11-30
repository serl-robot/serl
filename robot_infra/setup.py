from setuptools import setup
setup(name='franka_env',
      version='0.0.1',
      packages=['franka_env'],
      install_requires=['gymnasium',
                        'pyrealsense2',
                        'opencv-python',
                        'pyquaternion',
                        'hidapi',
                        'pyyaml',
                        'rospkg',
                        'scipy',
                        'requests',
                        'Pillow',
                        'flask',
                        'defusedxml']
)