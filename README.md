# A kernel conditional two-sample test
The structure of this project is the following:

    .
    ├── figures                     # where figures will be stored (not part of the repo)
    ├── results                     # where results will be stored (not part of the repo)
    ├── src                         # Source files
    │   ├── experiments             # Main scripts for configuring and running our numerical experiments
    │   ├── expyro                  # Utility package for experiment management
    │   ├── figures                 # Scripts for generating figures
    │   ├── rkhs                    # Core implementation of kernel methods and statistical test
    ├── requirements.txt            # .txt-file with package specifications
    └── README.md

## Installation
Create an environment called ``conditional-test`` with Python 3.12.6, pull the content from this repo into the
environment, and install all needed packages with:

```bash
cd conditional-test
source <path/to/venv>/bin/activate
pip install -r requirements.txt
```

## Reproducing numerical results
All of our numerical experiments can be reproduced from the command line. By running the following commands, you can
reproduce our results using the configurations used in the paper.

The outcome of every run is saved to `./results` from where they can be used to reproduce our figures stored under
`./figures`. 

### Illustrative Example
```bash
python -m src.experiments.example_1d
python -m src.figures.example_1d
```

### Empirical error rates
We repeat all experiments on empirical error rates for 100 different random seeds. You can adjust this as needed.
```bash
for i in $(seq 0 99);
do
    # for Figure 2 (left and middle)
    python -m src.experiments.error_rates disturbance --test-name="hu-lei__hg" --relative-norm=0.05 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="hu-lei__hg" --relative-norm=0.1 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="hu-lei__hg" --relative-norm=0.25 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="hu-lei__hg" --relative-norm=0.5 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="hu-lei__hg" --relative-norm=1 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="bootstrap" --relative-norm=0.05 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="bootstrap" --relative-norm=0.1 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="bootstrap" --relative-norm=0.25 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="bootstrap" --relative-norm=0.5 --seed=$i
    python -m src.experiments.error_rates disturbance --test-name="bootstrap" --relative-norm=1 --seed=$i
    
    # for Figure 2 (right)
    python -m src.experiments.error_rates local-disturbance --test-name="hu-lei__gt" --relative-norm=1 --weight=0.005 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="hu-lei__gt" --relative-norm=1 --weight=0.01 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="hu-lei__gt" --relative-norm=1 --weight=0.02 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="hu-lei__gt" --relative-norm=1 --weight=0.03 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="hu-lei__gt" --relative-norm=1 --weight=0.04 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="bootstrap" --relative-norm=1 --weight=0.005 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="bootstrap" --relative-norm=1 --weight=0.01 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="bootstrap" --relative-norm=1 --weight=0.02 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="bootstrap" --relative-norm=1 --weight=0.03 --tolerance=0.01 --seed=$i
    python -m src.experiments.error_rates local-disturbance --test-name="bootstrap" --relative-norm=1 --weight=0.04 --tolerance=0.01 --seed=$i
    
    # for Figure 3
    python -m src.experiments.error_rates mixture-noise --kernel-type="gaussian" --noise-mean=0.05 --seed=$i
    python -m src.experiments.error_rates mixture-noise --kernel-type="gaussian" --noise-mean=0.075 --seed=$i
    python -m src.experiments.error_rates mixture-noise --kernel-type="gaussian" --noise-mean=0.1 --seed=$i
    python -m src.experiments.error_rates mixture-noise --kernel-type="linear" --noise-mean=0.05 --seed=$i
    python -m src.experiments.error_rates mixture-noise --kernel-type="linear" --noise-mean=0.075 --seed=$i
    python -m src.experiments.error_rates mixture-noise --kernel-type="linear" --noise-mean=0.1 --seed=$i
    
    # for Figure 4
    python -m src.experiments.error_rates output-kernel --kernel-type="gaussian" --kernel-parameter=0.05 --seed=$i
    python -m src.experiments.error_rates output-kernel --kernel-type="gaussian" --kernel-parameter=0.1 --seed=$i
    python -m src.experiments.error_rates output-kernel --kernel-type="gaussian" --kernel-parameter=0.15 --seed=$i
    python -m src.experiments.error_rates output-kernel --kernel-type="polynomial" --kernel-parameter=1 --seed=$i
    python -m src.experiments.error_rates output-kernel --kernel-type="polynomial" --kernel-parameter=2 --seed=$i
    python -m src.experiments.error_rates output-kernel --kernel-type="polynomial" --kernel-parameter=3 --seed=$i
    
    # for Figure 6
    python -m src.experiments.error_rates dataset-size --test-name="bootstrap" --size=20 --seed=$i
    python -m src.experiments.error_rates dataset-size --test-name="bootstrap" --size=50 --seed=$i
    python -m src.experiments.error_rates dataset-size --test-name="bootstrap" --size=100 --seed=$i
    python -m src.experiments.error_rates dataset-size --test-name="bootstrap" --size=250 --seed=$i
done
```

You can create the corresponding figures (selectively) from the following commands.

```bash
python -m src.figures.error_rates --figure=2
python -m src.figures.error_rates --figure=3
python -m src.figures.error_rates --figure=4
python -m src.figures.error_rates --figure=6
```

### System monitoring
We repeat our system monitoring tutorial experiment for 50 different random seeds. You can adjust this as needed.

```bash
for i in $(seq 0 49);
do
    python -m src.experiments.monitoring --dimension=2 --disturbance=0.1 --seed=$i
    python -m src.experiments.monitoring --dimension=4 --disturbance=0.1 --seed=$i
    python -m src.experiments.monitoring --dimension=8 --disturbance=0.1 --seed=$i
    python -m src.experiments.monitoring --dimension=16 --disturbance=0.1 --seed=$i
    python -m src.experiments.monitoring --dimension=16 --disturbance=0.5 --seed=$i
    python -m src.experiments.monitoring --dimension=16 --disturbance=0.75 --seed=$i
done

python -m src.figures.monitoring
```
