<div align="center">

# Spaceship Titanic

A machine learning solution for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition. The task is to predict which passengers were transported to an alternate dimension when the ship collided with a spacetime anomaly.
</div>
I built this as a proper portfolio project, not just a quick Kaggle submission. The focus was on doing the data science correctly — clean validation, no leakage, decisions backed by ablation rather than guesswork.

---

## What's in here

The notebook covers the full pipeline from raw data to submission file. The main technical decisions I made:

**Leakage-free preprocessing** — I wrap all imputation and feature engineering inside a function that fits on the training fold and transforms the validation fold separately. A lot of Kaggle notebooks concatenate train and test before computing group sizes or filling missing values, which leaks information into CV. This doesn't.

**GroupKFold validation** — Passengers travelling together share a group ID in the data. If you split them randomly across folds, the model learns group-level patterns it wouldn't have at test time and your CV score is inflated. GroupKFold keeps each group in one fold only.

**Optuna with pruning** — 30 trials searching LightGBM's hyperparameter space. I originally ran 100 and saw the score plateau around trial 25, so 30 is plenty here. I wrote a custom pruning callback instead of using the Optuna-LightGBM integration because the integration had a metric direction conflict that was crashing every trial.

**LightGBM + CatBoost ensemble** — I tested XGBoost as a third model and ran the ablation properly (same folds, same preprocessing). It made the ensemble worse on this dataset, so I dropped it. Both remaining models use native categorical handling — no LabelEncoder.

**Threshold tuning** — Default 0.5 threshold assumes perfectly calibrated probabilities. I sweep from 0.30 to 0.70 on OOF predictions and pick the cutoff that actually maximises accuracy.

**SHAP** — Computed on the training set only. The explainer never sees test data.

---

## Results

Final CV accuracy (GroupKFold, 5 folds): ~0.817 for the ensemble. CatBoost and LightGBM individually sit around 0.814-0.815.

![Model Comparison](assets/model_comparison.png)

![SHAP Summary](assets/shap_summary.png)

The two biggest drivers are `CryoSleep` (passengers in cryo almost always got transported) and `TotalSpend` (high spenders were awake and much less likely to be transported). Not surprising when you think about it.

![Feature Importance](assets/feature_importance.png)

---

## Setup

```bash
git clone https://github.com/SyedMuhammadMujtabaKhalid/spaceship-titanic-ml-pipeline.git
cd spaceship-titanic-ml-pipeline
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and put them in the `data/` folder.

Then open the notebook and run all cells top to bottom:

```bash
jupyter notebook notebooks/spaceship-titanic-ml-pipeline.ipynb
```

It'll run the Optuna search, train the ensemble, and write a `submission.csv` to `outputs/`.

---

## Repo structure

```
spaceship-titanic-ml-pipeline/
├── assets/               # charts for the README
├── data/                 # train.csv, test.csv (not committed)
├── notebooks/
│   └── spaceship-titanic-ml-pipeline.ipynb
├── outputs/              # submission.csv lands here (not committed)
├── requirements.txt
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

---

## A few things I'd try next

Pseudo-labelling is the obvious next step — take test predictions the model is very confident about and add them to training. It's a bit rough but it genuinely helps on Kaggle. After that I'd look at stacking: train a simple meta-learner on the OOF probability vectors from each model instead of just averaging them.

The feature set is deliberately minimal right now. I tested age bins, log transforms, spending ratios, and a bunch of interaction terms. None of them moved the CV score, so they're all gone.

---

## Dataset overview

![Target Distribution](assets/target_distribution.png)

![Correlation Heatmap](assets/correlation_heatmap.png)

~8,700 training passengers, ~4,300 test. Target is nearly balanced at 50.4% transported, so no resampling needed. Missing values appear in almost every column which makes the imputation strategy non-trivial.

---

Made by [Syed Muhammad Mujtaba Khalid](https://github.com/SyedMuhammadMujtabaKhalid)
