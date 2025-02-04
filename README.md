# A Conditional Two-sample Test

The structure of this repository is the following:

    .
    ├── results                     # result files (empty)
    ├── src                         # Source files
    │   ├── experiments             # Main scripts for experiments
    │   ├── expyro                  # Utility package for experiment management
    │   ├── figures                 # Scripts for generating figures
    │   ├── rkhs                    # Utility package for kernel computations
    ├── requirements.txt            # .txt-file with package specifications
    └── README.md

## Reproducing results
You can reproduce all presented results with our hyperparameter configurations by running the following commands:

```bash
pip install -r requirements.txt
python -m src.experiments.error_rates
python -m src.experiments.example_1d
python -m src.experiments.example_2d
python -m src.experiments.monitoring
```

The results are saved to the `results` folder.

Our figures can be generated from these results by running the following commands:

```bash
python -m src.figures.figure_1
python -m src.figures.figure_2
python -m src.figures.figure_3
python -m src.figures.figure_4
python -m src.figures.figure_5
python -m src.figures.figure_6
python -m src.figures.figures_error_rates
```
