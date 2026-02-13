# Contributing to hybridlane

First off, thank you for considering contributing to hybridlane! We really appreciate
it!

## Submitting a Pull Request

All contributions should be submitted as a pull request on GitHub.

1.  Fork the repository and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  If you've changed APIs, update the documentation.
4.  Ensure the test suite passes.
5.  Make sure your code lints.
6.  Issue that pull request!

## Code Quality

This project uses pre-commit hooks to enforce code style and quality. Before you commit, please make sure to install and run the hooks.

```bash
uv tool install pre-commit --with pre-commit-uv
pre-commit install
```

Now the hooks will be run automatically before each commit.

## Commit Messages

We prefer that commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps us to automatically generate changelogs and to better understand the history of the project.

For example:

```
feat: Add a new feature
fix: Fix a bug
docs: Update documentation
```

## Licensing

All contributions to this project will be licensed under the terms of the project's existing license. A copy of the license can be found in the `LICENSE.txt` file in the root of the repository.
