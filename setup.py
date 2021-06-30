import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="es2hfa",
    version="0.0.1",
    author="Nandeeka Nayak",
    author_email="ndnayak2@illinois.edu",
    description="A compiler from a YAML description to HFA code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FPSG-UIUC/hfa-compiler",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    install_requires=[req for req in requirements if req[:2] != "# "],
    python_requires=">=3.6",
)
