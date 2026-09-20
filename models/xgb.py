import random
import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


random_state = 5

np.random.seed(random_state)
random.seed(random_state)

xgb_device = "cuda"

print("XGBoost version:", xgb.__version__)
print("Using XGBoost device:", xgb_device)


#load data
X = pd.read_csv("/home/cai/naomi/immuneproject/Xravi.csv")
Y = pd.read_csv("/home/cai/naomi/immuneproject/Yravi.csv")

#keep these for the feature importance file
gene_names = X["Name"].values
gene_descriptions = X["Description"].values

X = X.drop(columns=["Name", "Description"])

#samples as rows, genes as columns
X = X.T
X.index.name = "SampleID"

Y = Y.set_index("Harmonized_SU2C_RNA_Tumor_Sample_ID_v2")

#make sure rows line up with labels
X = X.loc[Y.index]

#no log transform or scaling needed, trees only care about the order of the values
X = X.astype(np.float32).values
y = Y["class_label"].values.astype(np.int64)

n_class_0 = np.sum(y == 0)
n_class_1 = np.sum(y == 1)

print("Samples:", X.shape[0])
print("Genes:", X.shape[1])
print("Class 0:", n_class_0)
print("Class 1:", n_class_1)

if len(np.unique(y)) != 2:
    raise ValueError("The dataset must contain exactly two classes.")

#need enough samples in the smaller class for the nested 5 fold cv
if min(n_class_0, n_class_1) < 10:
    raise ValueError("The smallest class has fewer than 10 samples.")

#class 0 / class 1, used as one of the class weight options
base_scale_pos_weight = n_class_0 / n_class_1

print(f"Class 0 / Class 1: {base_scale_pos_weight:.4f}")


#random search settings
n_estimators_options = [100, 200, 300, 500]
max_depth_options = [2, 3, 4, 5]
gamma_options = [0, 0.1]
subsample_options = [0.8, 1.0]

#lots of genes and few samples, so let each tree see a small part of the genes
colsample_bytree_options = [0.05, 0.1, 0.3, 0.5]

#stronger regularization for the small sample size
min_child_weight_options = [1, 3, 5]
reg_lambda_options = [1, 5, 10]

#1.0 is no class weighting, the others weight the smaller class more
scale_pos_weight_options = sorted(
    set([1.0, 1.5, round(base_scale_pos_weight, 3), 2.0])
)

#learning rate is sampled on a log scale
lr_range = (0.01, 0.1)

param_names = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "gamma",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
    "scale_pos_weight"
]

#how many random configurations to try in each search
n_random_trials = 50


def log_uniform(rng, low, high):
    value = 10 ** rng.uniform(np.log10(low), np.log10(high))
    #keep 3 significant digits so it prints nicely
    return float(f"{value:.3g}")


def sample_params(rng):
    #randomly pick a full set of hyperparameters

    return {
        "n_estimators": rng.choice(n_estimators_options),
        "max_depth": rng.choice(max_depth_options),
        "learning_rate": log_uniform(rng, *lr_range),
        "gamma": rng.choice(gamma_options),
        "subsample": rng.choice(subsample_options),
        "colsample_bytree": rng.choice(colsample_bytree_options),
        "min_child_weight": rng.choice(min_child_weight_options),
        "reg_lambda": rng.choice(reg_lambda_options),
        "scale_pos_weight": rng.choice(scale_pos_weight_options)
    }


def make_model(params):
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=params["scale_pos_weight"],
        objective="binary:logistic",
        eval_metric="logloss",
        importance_type="gain",
        tree_method="hist",
        device=xgb_device,
        random_state=random_state,
        n_jobs=-1
    )


def get_auc(params, X_train, y_train, X_val, y_val):
    #train on the training data and return the auc on the val data

    model = make_model(params)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y_val, probabilities)


print("Total scale_pos_weight options:", scale_pos_weight_options)
print("Random trials per search:", n_random_trials)


#outer 5 fold cv estimates performance
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

#inner 5 fold cv picks hyperparameters
inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

outer_auc_scores = []
outer_results = []
best_params_per_outer_fold = []
outer_importances = []


for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(
        outer_skf.split(X, y),
        start=1
):

    print("\n" + "=" * 60)
    print(f"Outer fold {outer_fold}/5")
    print("=" * 60)

    X_outer_train = X[outer_train_idx]
    y_outer_train = y[outer_train_idx]

    X_outer_val = X[outer_val_idx]
    y_outer_val = y[outer_val_idx]

    print("Outer training samples:", len(y_outer_train))
    print("Outer validation samples:", len(y_outer_val))

    #new random configurations for this fold
    fold_rng = random.Random(random_state + outer_fold)

    inner_results = []

    for trial_number in range(1, n_random_trials + 1):

        params = sample_params(fold_rng)

        print(f"\nTrial {trial_number}/{n_random_trials}")
        print("Testing:", params)

        inner_auc_scores = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_skf.split(X_outer_train, y_outer_train),
                start=1
        ):

            auc = get_auc(
                params,
                X_outer_train[inner_train_idx],
                y_outer_train[inner_train_idx],
                X_outer_train[inner_val_idx],
                y_outer_train[inner_val_idx]
            )

            inner_auc_scores.append(auc)

            print(f"  Inner fold {inner_fold} AUC: {auc:.4f}")

        mean_inner_auc = float(np.mean(inner_auc_scores))
        std_inner_auc = float(np.std(inner_auc_scores, ddof=1))

        inner_results.append({
            **params,
            "mean_inner_auc": mean_inner_auc,
            "std_inner_auc": std_inner_auc
        })

        print(f"  Mean inner AUC: {mean_inner_auc:.4f} +/- {std_inner_auc:.4f}")

    #best configuration first
    inner_results.sort(key=lambda r: r["mean_inner_auc"], reverse=True)

    pd.DataFrame(inner_results).to_csv(
        f"XGBoost_OuterFold_{outer_fold}_RandomSearch_Results.csv",
        index=False
    )

    #best hyperparameters, chosen with inner cv only
    best = inner_results[0]

    best_params = {name: best[name] for name in param_names}
    best_inner_auc = best["mean_inner_auc"]

    best_params_per_outer_fold.append({
        "outer_fold": outer_fold,
        **best_params,
        "inner_cv_auc": best_inner_auc
    })

    print("\nBest parameters:")
    print(best_params)
    print(f"Best inner CV AUC: {best_inner_auc:.4f}")

    #train on the whole outer training set with the best parameters
    print(f"\nTraining best model for outer fold {outer_fold}...")

    final_outer_model = make_model(best_params)
    final_outer_model.fit(X_outer_train, y_outer_train)

    #evaluate once on the outer validation set
    probabilities = final_outer_model.predict_proba(X_outer_val)[:, 1]

    outer_auc = roc_auc_score(y_outer_val, probabilities)

    outer_auc_scores.append(outer_auc)

    #keep the gene importances of this fold's model
    outer_importances.append(final_outer_model.feature_importances_)

    outer_results.append({
        "outer_fold": outer_fold,
        "outer_auc": outer_auc,
        "inner_cv_auc": best_inner_auc,
        **best_params
    })

    print(f"\nOuter fold {outer_fold} AUC: {outer_auc:.4f}")


#save outer fold results
pd.DataFrame(outer_results).to_csv(
    "XGBoost_NestedCV_Outer_Results.csv",
    index=False
)

pd.DataFrame(best_params_per_outer_fold).to_csv(
    "XGBoost_NestedCV_Selected_Parameters.csv",
    index=False
)

nested_mean_auc = float(np.mean(outer_auc_scores))
nested_std_auc = float(np.std(outer_auc_scores, ddof=1))

print("\n" + "=" * 60)
print("5-fold nested cross-validation results")
print("=" * 60)

for i, auc in enumerate(outer_auc_scores, start=1):
    print(f"Outer fold {i} AUC: {auc:.4f}")

print(f"\nNested CV AUC: {nested_mean_auc:.4f} +/- {nested_std_auc:.4f}")


#final hyperparameter search on all the data
#same 5 fold search as the inner cv, this only picks the settings for the saved model
#it is not part of the nested cv estimate
print("\n" + "=" * 60)
print("Final hyperparameter search")
print("=" * 60)

final_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

final_rng = random.Random(random_state + 100)

final_search_results = []

for trial_number in range(1, n_random_trials + 1):

    params = sample_params(final_rng)

    print(f"\nFinal trial {trial_number}/{n_random_trials}")
    print(params)

    fold_auc_scores = []

    for train_idx, val_idx in final_skf.split(X, y):

        auc = get_auc(
            params,
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx]
        )

        fold_auc_scores.append(auc)

    mean_auc = float(np.mean(fold_auc_scores))
    std_auc = float(np.std(fold_auc_scores, ddof=1))

    final_search_results.append({
        **params,
        "mean_auc": mean_auc,
        "std_auc": std_auc
    })

    print(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")


final_search_results.sort(key=lambda r: r["mean_auc"], reverse=True)

pd.DataFrame(final_search_results).to_csv(
    "XGBoost_Final_Hyperparameter_Search.csv",
    index=False
)

final_best = final_search_results[0]

final_params = {name: final_best[name] for name in param_names}

print("\nFinal hyperparameters:")
print(final_params)


#train the final model on all the data
#this one is for future predictions, the nested cv auc is the performance estimate
print("\n" + "=" * 60)
print("Training final model on all data")
print("=" * 60)

final_model = make_model(final_params)
final_model.fit(X, y)

final_model_path = "final_XGBoost_nestedCV_5fold_model.json"
final_model.save_model(final_model_path)

pd.DataFrame([final_params]).to_csv(
    "final_XGBoost_nestedCV_5fold_params.csv",
    index=False
)


#feature importances (gain)
#the final model is trained on everything, the outer fold models show how stable it is
outer_importances = np.array(outer_importances)

importance_df = pd.DataFrame({
    "Name": gene_names,
    "Description": gene_descriptions,
    "final_model_importance": final_model.feature_importances_,
    "mean_outer_fold_importance": outer_importances.mean(axis=0),
    "outer_folds_used": (outer_importances > 0).sum(axis=0)
})

importance_df = importance_df.sort_values(
    by="final_model_importance",
    ascending=False
).reset_index(drop=True)

importance_df.to_csv("XGBoost_Feature_Importance.csv", index=False)

print("\nTop 20 genes by importance (gain):")
print(importance_df.head(20).to_string())

print("\n" + "=" * 60)
print("Final model saved:", final_model_path)
print("Feature importances saved: XGBoost_Feature_Importance.csv")
print(f"Nested CV AUC: {nested_mean_auc:.4f} +/- {nested_std_auc:.4f}")
print("\nGPU used:", xgb_device)