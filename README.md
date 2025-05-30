# SBDEPS CNN Model
A CNN Model trained for Satellite-Based Detection of Electric Parking Stations, This code was created for a Deep Learning Project in 2025.


## To validate on a labeled custom dataset...

1. Save labeled images in "eparking" folder (y = with Charging Station, n = without)
2. Open and run main.ipynb




### Files:

cnn_optimised_sgd.keras                     - the best performing model, usable on satellite images (250x250px)
main.ipynb                                  - validates model on custom dataset
model_summary.ipynb                         - provides insight to a model's architecture
cnn_model_performances.ipynb                - tests and compares models on the entire dataset
cnn_model_test_performance.ipynb            - tests a Model's accuracy and F1score on the test-dataset
cnn_parameter_tuning_optuna.ipynb           - runs iterations to find optimal parameter settings
cnn_v_1.ipynb                               - code used to train baseline model
cnn_v_2.ipynb                               - code used to train manually fine-tuned model

| File Name                              | Description |
|----------------------------------------|-------------|
| `cnn_optimised_sgd.keras`              | The best performing model, usable on satellite images (250x250 px) |
| `main.ipynb`                           | Validates model on a custom dataset |
| `model_summary.ipynb`                  | Provides insight into a model's architecture |
| `cnn_model_performances.ipynb`         | Tests and compares models on the full dataset |
| `cnn_model_test_performance.ipynb`     | Evaluates model accuracy and F1 score on the test dataset |
| `cnn_parameter_tuning_optuna.ipynb`    | Runs Optuna to find optimal parameter settings |
| `cnn_v_1.ipynb`                        | Code used to train the baseline model |
| `cnn_v_2.ipynb`                        | Code used to train the manually fine-tuned model |

---

### Folders:

eparking                                    - custom folder to evaluate the model with
__archived                                  - contains old codes and old models
__misclassified_images                      - contains False Positives and False Negatives of select Models
data                                        - contains entire dataset
data_split                                  - contains Train-Test data (80% - 20%)
models                                      - contains a selection of high performing models





