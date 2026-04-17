from setuptools import setup, find_packages

setup(
    name="MREF",
    version="0.1.0",
    description="MREF: A Multi-Aspect Reference-Free Evaluation Framework for Text Simplification",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AlMotasem Bellah Al Ajlouni, Jinlong Li, Huanhuan Chen",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "spacy",
        "wordfreq",
        "nltk",
    ],
    entry_points={
        "console_scripts": [
            "mref=MREF.cli:main",
        ],
    },
    python_requires=">=3.10",
)
