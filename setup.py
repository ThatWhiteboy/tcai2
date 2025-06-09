from setuptools import setup, find_packages

setup(
    name="tcai2",
    version="0.1.0",
    description="AI Utilities Library",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    python_requires=">=3.8",
)