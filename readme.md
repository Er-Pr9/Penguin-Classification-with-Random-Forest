# Penguin Species Classification with Random Forests

A concise, reproducible project that trains and evaluates a Random Forest classifier to predict the penguin species (Adélie, Chinstrap, Gentoo) from physical measurements using the Palmer Penguins dataset. The workflow is implemented in a Jupyter notebook and includes data loading, preprocessing, modeling, and evaluation with common metrics and visual diagnostics.

### Project goals

- Build a supervised learning pipeline to classify penguin species from culmen dimensions, flipper length, body mass, and limited categorical context.

- Explore how changing Random Forest hyperparameters affects accuracy and confusion matrix outcomes to support robust field classification.


## Dataset

- Primary CSV: penguins_size.csv (simplified Palmer Penguins with columns: species, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, island, sex).

- Target: species ∈ {Adelie, Chinstrap, Gentoo}.

- Features: numeric (culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g) and categorical (island, sex).



## Methods

- Model: **Random Forest** classifier with experiments across key hyperparameters (e.g., n_estimators, max_features, bootstrap) to study performance trade-offs.

- Tuning: GridSearchCV is used to search over a small grid of tree and feature parameters for improved accuracy and stability.

- Evaluation: accuracy_score, classification_report, confusion_matrix, and ConfusionMatrixDisplay for interpretable results.


## Repository structure

- Penguin_project.ipynb — end-to-end notebook with data loading, EDA, model training, and evaluation.

- penguins_size.csv — dataset file expected by the notebook (place at project root or update the path in the notebook).


Project Structure:

```
.
├── Penguin_project.ipynb
├── penguins_size.csv
```


## Setup

- Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn.

- Install with pip:

```
pip install numpy pandas matplotlib seaborn scikit-learn
```


## Usage

- Open the notebook: Penguin_project.ipynb in Jupyter (Notebook or Lab).

- Ensure penguins_size.csv is present at the path expected by the cell df = pd.read_csv("penguins_size.csv") or adjust the path in that cell before running.

- Run cells sequentially to execute EDA, preprocessing, model training, and evaluation; the notebook demonstrates baseline and tuned Random Forest configurations.


## Modeling details

- Baseline: the notebook includes runs such as RandomForestClassifier(n_estimators=10, random_state=101) to establish initial performance.

- Hyperparameters studied: n_estimators, max_features, bootstrap; a GridSearchCV sweep is demonstrated to choose better configurations.

- Train/validation split: train_test_split is used to create held-out evaluation data for fair metrics and plots.


## Evaluation

- Metrics: accuracy_score and classification_report summarize overall and per-class performance, while confusion_matrix and ConfusionMatrixDisplay reveal misclassification patterns among species.

- Plots: confusion matrix visualization helps diagnose class confusions for model iteration and field utility considerations.


## Reproducibility

- Random states: example runs set random_state=101 for deterministic behavior in training and splitting where applicable, aiding reproducibility across executions and environments.

- Grid search: all parameter grids and scoring choices are defined in-notebook for transparent model selection and reruns.





