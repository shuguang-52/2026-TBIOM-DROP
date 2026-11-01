"""Install DROP's Torchreid runtime; optional Cython ranking acceleration."""
import os
import re
from pathlib import Path
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
version = re.search(r"__version__ = '([^']+)'", (ROOT / "torchreid/__init__.py").read_text()).group(1)
ext_modules = []
if os.environ.get("DROP_BUILD_CYTHON") == "1":
    import numpy
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension(
        "torchreid.metrics.rank_cylib.rank_cy",
        ["torchreid/metrics/rank_cylib/rank_cy.pyx"],
        include_dirs=[numpy.get_include()],
    )], language_level=3)

setup(
    name="torchreid",
    version=version,
    description="DROP: task-specific features for occluded person re-identification",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/shuguang-52/2026-TBIOM-DROP",
    license="See LICENSE and THIRD_PARTY_NOTICES.md",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")],
    ext_modules=ext_modules,
)
