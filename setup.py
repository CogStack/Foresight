import setuptools

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foresight",
    version="0.5",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Deep Generative Modelling of Patient Timelines using Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/Foresight",
    packages=[
        "foresight",
        "foresight.datasets",
        "foresight.metrics",
        "foresight.utils",
        "foresight.tokenizers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "datasets>=2.15",
        "transformers>=4.35",
        "torch",
        "numpy",
    ],
    extras_require={
        "flash-attn": ["flash-attn>=2.3"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
