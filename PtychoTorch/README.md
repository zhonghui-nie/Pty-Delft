# PRA ( readme )

**Phase Retrieval Algorithm** - an optimized python implementation to run iterative ptychography-algorithms to retrieve phases of measured diffraction-patterns.

## Goals

- Implementation of multiple different ptychography-based algorithms
- **Fast**: RPA uses pytorch as backend. Pytorch has proven to run fourier transformations the [fastest]( https://thomasaarholt.github.io/fftspeedtest/fftspeedtest.html ) among different frameworks ( numpy, cupy, tensorflow ). In pytorch, calculations can easily be performed on both CPU or CUDA.
- **Modular**: PRA is structured in a highly modular manner. Classes are inherited from abstract base-classes. This allows straightforward implementation of custom classes (e.g. a custom implementation of an optimizer).
- **Hackable**: PRA is designed with priority on flexibilty. Users have direct access to intermediate steps during reconstruction. This can be helpful to gain detailed insights of the training, or to add extra steps to quickly test out variations of reconstruction algorithms.

## Getting started

### Installation with `uv` (Recommended)

Requirements:
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- git

#### Installation PRA (Development)

CPU:
```bash
git clone https://git.amolf.nl/EUVGI/PRA
cd PRA
uv sync --extra cpu --extra dev
uv run pre-commit install
```

CUDA 12.1 support (GPU)
```bash
git clone https://git.amolf.nl/EUVGI/PRA
cd PtychoTorch
uv sync --extra cu121 --dev
uv pip install -e .
uv run pre-commit install
```

#### Run tests

```bash
uv run pytest
```

#### Build and serve docs

```bash
uv run sphinx-build doc doc/_build
uv run python -m http.server --directory doc/_build
```

### Installation with `pip`

#### Create and Activate New Virtual Environment

Make sure you use python version 3.10 or greater.

Windows:
```bash
python3 -m venv venv
.\venv\Scripts\activate
```

Linux/Mac:
```bash
python3 -m venv venv
source .\venv\bin\activate
```

Then **install pytorch** following the official instructions https://pytorch.org/get-started.
If you skip this step, then pytorch will only be installed with CPU-support!

#### Installation PRA (As External Package)

```bash
pip install git+https://git.amolf.nl/EUVGI/PRA
```

#### Installation PRA (Development)

Requires `pip >= 21.3` for editable installation.
Make sure you activated your virtual environment.

```bash
git clone https://git.amolf.nl/EUVGI/PRA
pip install -e PRA[dev]
cd PRA
pre-commit install
```

#### Build and Serve Documentation

From within the PRA-folder, run

```bash
sphinx-build doc doc/_build
```

to build (or update) the documentation.
If installed on a local machine, then the built documentations can be accessed by opening `doc/_build/index.html`.

In case the documentations were installed on a remote computer, then a http-server serving the documentation can be started by
```bash
python -m http.server --directory doc/_build
```
which can be accessed by any computer within the network by opening <http://{IP_OF_REMOTE_COMPUTER}:8000>.


### First Steps

Check out the examples in the Scripts-folder.
So far implemented:

- ePIE
- aPIE
- multiPIE

## Contribute

- Create branches and open pull requests to add new features
- Style is enforced by `black​`, `flake8`, ect. These programs are hooked to the `pre-commit`. You can check the style by running `pre-commit run --all-files`

## Roadmap

- [x] implementation of ePIE
- [x] implementation of mPIE
- [x] implementation of aPIE
- [x] implementation of AutoGradPIE
- [x] implementation of multiPIE
- [ ] implementation of zPIE (using gradient descent)
- [ ] parallized reconstruction on cluster / multiple computers (e.g. via [Ray](https://www.ray.io/)?)
