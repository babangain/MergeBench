# Implementation of model merging methods

## Merging
Each merging algorithms contain specific parameters and configurations, and all of which are prepared in `prepare_args.py`. Please refer to their original papers for more detailed definitions. We provide examples for running each merging algorithms in `scripts/merge.sh`.

To install the packages required for merging, run the following commands:
```
conda create -n merging
conda activate merging
pip install -r requirements.txt
```

## Adding new merging methods
1. Create a new python file under the `merging_methods` directory, e.g., `task_arithmetic.py`. 
2. Within the file, define the class name for the merging algorithm `TaskArithmetic`, which inherits from the abstract method `Merger` in `merger.py`. 
3. Remember to add the class in `__init__.py`, e.g., 
```
from merging_methods.task_arithmetic import TaskArithmetic
```
4. Add any method dependent hyperparameters in `prepare_args.py`, which will be passed into the class via `kwargs`. The argument parsing in `main.py` only handles generic arguments.
5. Overwrite the `merge` function to implement the details of merging, which ends with saving the merged model and tokenizer. 
