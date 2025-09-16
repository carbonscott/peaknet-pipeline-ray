from setuptools import setup, find_packages

setup(
    name="peaknet-pipeline-ray",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "ray[default]>=2.0.0",
        "pyyaml>=5.4.0",
        "numpy>=1.20.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        'console_scripts': [
            'peaknet-pipeline=peaknet_pipeline_ray.cli:main',
        ],
    },
    python_requires=">=3.8",
    author="PeakNet Pipeline Team",
    description="Scalable PeakNet ML inference pipeline with Ray",
    long_description_content_type="text/markdown",
)