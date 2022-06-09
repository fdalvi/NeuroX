
# Contributing to NeuroX

Welcome, and thank you for your interest in improving NeuroX. This document provides a high-level overview of the contribution process. For any clarifications and/or concerns, feel free to open an issue in the GitHub issue tracker.

NeuroX is a toolkit which is evolving and growing quite quickly, so it is important to share your intentions before you start working on something. This is to make sure that your effort is not being duplicated elsewhere. We expect contributions to be of several kinds like:

1. Bug fixes (existing in the tracker or new)
2. Quality-of-Life Improvements (this may mean better wrappers around existing code, better documentation, improved/additional tests, example notebooks)
3. New algorithms for neuron discovery and analysis (usually an implementation of an existing paper)
4. New general algorithms (for things like processing data, filtering, balancing, performing analysis, evaluation, etc)

There may be other types of contributions that we may have missed, so feel free to open an issue. For an idea within the above categorization, please mention the category in your issue, along with supporting code/documents such as a minimal reproducible example for bug fixes, or links to papers for new algorithms. Once you have received a go-ahead, implement your feature, document your code and add tests before submitting a pull request to the repository.


## Development Process

### Setting up the repository and dependencies

Clone the repository, create a virtual enviroment and install all the development dependencies using the following commands:

```bash
git clone https://github.com/fdalvi/NeuroX.git
python -m venv .neurox-dev
source .neurox-dev/bin/activate
pip install -e '.[dev]'
```

### Code Style

NeuroX aims to keep a consistent style, and is enforcing `ufmt` for all code in the repository. Run

```bash
./scripts/format_code.sh
```

to format all package and test code automatically.

### Unit Tests

The tests are run using `pytest` (with coverage support):
```bash
./scripts/run_tests.sh
```

You can also use python's built-in `unittest` module:

```bash
python -m unittest
```

### Update Version

NeuroX uses semantic version (major.minor.patch). To update version, use `bump2version`:
```bash
# increment patch part of the version
bump2version patch

# increment minor part of the version and tag, commit
bump2version --tag --commit minor
```

### Documentation
The following command will build the documentation locally and launch the page in your browser:
```bash
./scripts/generate_docs.sh
```

## Pull Requests
We actively welcome pull requests (and appreciate them to be pre-discussed in an issue).

1. Fork the repo and create your branch from `master`.
2. Implement your bugfix/feature in the new branch
3. Add tests for any new/modified code
4. Ensure the entire test suite passes
5. Add/Modify documentation for any new/modified code


## Issues

We use GitHub issues to track all bugs and discussion, please be as specific as possible in your issue.


## License

By contributing to NeuroX, you agree that your contributions will be licensed under BSD 3-Clause License.
