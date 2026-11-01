# Car Fuel Efficiency

## Problem

Predict a car's fuel efficiency in miles per gallon (`fuel_efficiency_mpg`) from its
specifications: model year, origin, fuel type, drivetrain, number of doors, engine
displacement, number of cylinders, horsepower, vehicle weight, and acceleration.

The goal is to build and compare tree-based regression models, tune their
hyperparameters, and understand which features drive fuel efficiency.

## Dataset

`data/car_fuel_efficiency_2026.csv` contains 10,000 rows and 11 columns. Missing
values are filled with zeros. The data is split into 60% train / 20% validation /
20% test using `random_state=1`, and categorical features are one-hot encoded with
`DictVectorizer(sparse=True)`.

## Solution

The analysis lives in `fuel_efficiency_tree_models.ipynb`, which covers:

1. A shallow decision tree (`max_depth=1`) to find the single most informative split.
2. A random forest baseline and its validation RMSE.
3. Tuning the number of trees (`n_estimators`).
4. Selecting tree depth (`max_depth`) by mean validation RMSE across forest sizes.
5. Extracting and aggregating feature importances.
6. Comparing XGBoost learning rates (`eta`) over 100 boosting rounds.

## Results

| Analysis | Outcome |
| --- | --- |
| First split (depth-1 tree) | `model_year` (threshold ~1997.5) |
| Random forest, 10 trees | validation RMSE 1.833 |
| Best `n_estimators` | 150 (RMSE 1.769) |
| Best `max_depth` by mean RMSE | 10 (mean RMSE 1.754) |
| Most important candidate feature | `vehicle_weight` (importance 0.203) |
| Most important feature overall | `model_year` (importance 0.296) |
| XGBoost `eta=0.3` vs `eta=0.1` | 1.827 vs **1.725** (eta 0.1 wins) |

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run jupyter lab
```

Open `fuel_efficiency_tree_models.ipynb` and run all cells.

## Project structure

```
.
├── data/
│   └── car_fuel_efficiency_2026.csv
├── fuel_efficiency_tree_models.ipynb
├── main.py
├── pyproject.toml
└── uv.lock
```
