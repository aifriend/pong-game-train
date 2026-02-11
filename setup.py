"""Setup script for Pong PPO Training Environment."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pong-ppo-env",
    version="2.0.0",
    author="Pong RL Contributors",
    description="Fast Pong RL environment optimized for PPO training with win-focused rewards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pong-ppo-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pong-train=scripts.train_ppo_curriculum:main",
            "pong-eval=scripts.evaluate_agent:main",
            "pong-play=scripts.play:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pong": ["resources/*"],
    },
)
