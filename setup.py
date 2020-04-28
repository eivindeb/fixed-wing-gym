from setuptools import setup

setup(
    name='gym-fixed-wing',
    version='0.1.0',
    url="https://github.com/eivindeb/fixed-wing-gym",
    author="Eivind Bøhn",
    author_email="eivind.bohn@gmail.com",
    description="OpenAI Gym wrapper for PyFly - Fixed-Wing Flight Simulator",
    packages=['gym_fixed_wing'],
    package_data={"gym_fixed_wing": ["fixed_wing_config.json"]},
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    install_requires=[
        "cycler>=0.10.0",
        "future>=0.17.1",
        "gym>=0.12.5",
        "kiwisolver>=1.1.0",
        "matplotlib>=3.1.0",
        "numpy>=1.16.4",
        "pyglet>=1.3.2",
        "pyparsing>=2.4.0",
        "python-dateutil>=2.8.0",
        "scipy>=1.3.0",
        "six>=1.12.0",
        "pyfly-fixed-wing==0.1.2"
    ]
)