# Optuna Logging in Comet.ml

This repo contains a couple of examples that show how to send [Optuna](https://github.com/optuna/optuna) study execution
logs to [Comet.ml](https://www.comet.ml).

## How-to
Install the requirements
```shell
pip install -r requirements.txt
```

Execute `python sklearn_simple.py` for in-the-script, explict comet_ml logging.

Execute `python sklearn_simple_comet.py` for transparent logging to comet_ml. See 
[https://github.com/thekovach/optuna/pull/1]() for implementation details. 
