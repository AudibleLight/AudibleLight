from setuptools import setup, find_packages

setup(
    name="audiblelight",
    version="0.1.0",
    description="A library for soundscape synthesis using spatial impulse responses derived from ray-traced room scans",
    author="Huw Cheston",
    author_email="h.cheston@qmul.ac.uk",
    url="",
    download_url="https://github.com/AudibleLight/AudibleLight/releases",
    packages=["audiblelight"],
    keywords="audio sound soundscape environmental ambisonics microphone array sound event detection localization",
    license="Creative Commons Attribution",
    classifiers=[
        "License :: Creative Commons Attribution 4.0",
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
    ],
    install_requires=find_packages(),
    extras_require={
        "test": [
            "coverage",
            "pytest",
            "black",
        ]
    },
)
