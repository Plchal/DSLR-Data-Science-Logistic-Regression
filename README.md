# DSLR: Data Science × Logistic Regression

*This project has been created as part of the 42 curriculum by pchalmin and tle-floc.*

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)

## Description

<!-- description -->

### Bonus
- More fields for describe [x]
- How to answer at : Which Hogwarts course has a homogeneous score distribution between all four houses ? [x]
- How to answer at : What are the two features that are similar ? [x]
- How to answer at : From this visualization, which features are you going to use for your logistic regression ? []
- Stochiastic Gradient Descent [x]
- Mini Batch [x]
- Soft - Max [x]
- Momentum []
- Adam []
- Plot cost [x]

## Tools

| Tool | Role |
| ---- | ---- |
| `python3` | Interpreter |
| `ruff format` | Formatter |
| `ruff check` | Linter |

## Instructions

### Build and run

Train the one-vs-all classifiers (default) or a single softmax classifier:

```bash
python3 src/logreg_train.py -f dataset_train.csv
python3 src/logreg_train.py -f dataset_train.csv --softmax
```

Hyperparameters and optimizer can be selected with
`--optimizer {batch,sgd,minibatch}`, `--alpha`, `--iteration`, `--batch_size`
and `--seed`. The weights are written to `weights.csv` (one-vs-all) or
`weights_softmax.csv` (`--softmax`), or to `--output <file>`.

The features used for the training can be restricted with `--features`,
which takes a comma-separated list of column names:

```bash
python3 src/logreg_train.py -f dataset_train.csv --features Astronomy,Herbology,Charms
```

By default every numeric column is used. The selected features are stored in
the weights file, so the prediction automatically uses the same ones.

### Predict

Generate the predictions with the one-vs-all weights (default) or with the
softmax weights:

```bash
python3 src/logreg_predict.py dataset_train.csv
python3 src/logreg_predict.py dataset_train.csv --softmax
```

The predictions are written to `houses.csv`, or to `--output <file>`. The
softmax prediction requires the weights to use the same features as the input
file.

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
