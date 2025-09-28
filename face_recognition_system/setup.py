from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="face-recognition-system",
    version="1.0.0",
    author="Face Recognition Team",
    author_email="team@facerecognition.com",
    description="A comprehensive face recognition system for access control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/face-recognition-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-recognition-train=src.scripts.train:main",
            "face-recognition-api=src.api.main:main",
        ],
    },
)