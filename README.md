# DSLR: Data Science × Logistic Regression

*This project has been created as part of the 42 curriculum by pchalmin and tle-floc.*

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)

## Description

<!-- description -->

### Bonus
- More fields for describe []
- How to answer at : Which Hogwarts course has a homogeneous score distribution between all four houses ? []
- How to answer at : What are the two features that are similar ? []
- How to answer at : From this visualization, which features are you going to use for your logistic regression ? []
- Stochiastic Gradient Descent []
- Mini Batch []
- Soft - Max []
- Momentum []
- Adam []
- Plot cost []

## Tools

| Tool | Role |
| ---- | ---- |
| `python3` | Interpreter |
| `ruff format` | Formatter |
| `ruff check` | Linter |

## Instructions

### Build and run

```bash
python3 -m src.main
```

### With Nix

[Nix](https://nixos.org/) is a tool of package management and system configuration to make reproducible, declarative and reliable systems.

To use the project package management and system configuration:
```bash
nix develop
```

To be able to use development tools as well (linter, formatter, commit message management, etc.):
```
nix develop .#dev
```

### Commit

This repository uses [Conventional Commits](https://conventionalcommits.org),
enforced by [Husky](https://typicode.github.io/husky/), [Commitizen](https://commitizen-tools.github.io/commitizen/) and [Commitlint](https://commitlint.js.org/).

```bash
npm run commit
```

The `pre-commit` hook rejects the commit when the formatter or the linter
reports an error on a staged file. The `commit-msg` hook rejects the commit
when the message does not follow the convention.

## Resources

[Pair Plot](https://www.geeksforgeeks.org/python/pairplot-in-matplotlib/)
