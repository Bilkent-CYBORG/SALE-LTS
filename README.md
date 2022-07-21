# Safe Leveling Linear Bandits

Author: Ilker Demirel ([ilkerd1997@gmail.com](mailto:ilkerd1997@gmail.com))

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## UVa/PADOVA T1DM simulator

We use the python implementation of the UVa/PADOVA simulator, which is available under MIT License at [simglucose](https://github.com/jxx123/simglucose). We obtained the simulator outputs for different patient, meal event, insulin intake triples and saved the results in the following directories,

```
/experiments/calc_res/
/experiments/calc_res_clinician_data/
```

## Running the experiments

We have two distinct notebooks for the experiments, one being for the clinician experiment.

```experiments
/experiments/clinician_comparison/clinician_comparison.ipynb
/experiments/MME/MME.ipynb
```

You can run these notebooks to reproduce the results, which will be saved under the same parent directory.

## Plotting the results

The necessary data to obtain the plots and the numerical results in the table is already available under the aforementioned directories. One can run the experiments as described before to reproduce the results. Finally, to obtain the plots and the numerical results in the table, it is sufficient to run a single noteboook in the following directory,

```
/experiments/plot.ipynb
```

