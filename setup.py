"""
Package setup for StudyFriend.

Author: Othmane Dardouri
"""

import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup

root_dir    = Path(__file__).parent
package_dir = root_dir / "study_friend"
sys.path.insert(0, str(package_dir))

requirements_path = root_dir / "requirements.txt"
with requirements_path.open() as fid:
    requirements = [l.strip() for l in fid if l.strip() and not l.startswith("#")]

if not torch.backends.mps.is_available():
    requirements = [r for r in requirements if not r.startswith("mlx-vlm")]
if not torch.cuda.is_available():
    requirements = [r for r in requirements if not r.startswith("bitsandbytes")]

setup(
    name="study_friend",
    version="2.0.0",
    description="Offline AI tools to generate study notes from PDF slides.",
    long_description=(root_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Othmane Dardouri",
    author_email="",
    url="https://github.com/OthmaneDardouri/study-friend",
    license="MIT",
    python_requires=">=3.10",
    install_requires=requirements,
    packages=find_packages(where=str(root_dir)),
    entry_points={
        "console_scripts": [
            "study_friend.query   = study_friend.query:main",
            "study_friend.convert = study_friend.convert:main",
            "study_friend.display = study_friend.display:main",
        ]
    },
)
