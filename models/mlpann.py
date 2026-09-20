import random
import copy
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


random_state = 5

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

#set to False to train on raw TPM instead
log_transform = True


#load data
X = pd.read_csv("/home/cai/naomi/immuneproject/Xravi.csv")
Y = pd.read_csv("/home/cai/naomi/immuneproject/Yravi.csv")

gene_names = X["Name"].values
X = X.drop(columns=["Name", "Description"])

#samples as rows, genes as columns
X = X.T
X.index.name = "SampleID"

Y = Y.set_index("Harmonized_SU2C_RNA_Tumor_Sample_ID_v2")

#make sure rows line up with labels
X = X.loc[Y.index]

X = X.astype(np.float32).values
y = Y["class_label"].values.astype(np.int64)

#TPM is really skewed so use log2(TPM + 1)
if log_transform:
    X = np.log2(X + 1)

print("Samples:", X.shape[0])
print("Genes:", X.shape[1])
print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))

if len(np.unique(y)) != 2:
    raise ValueError("The dataset must contain exactly two classes.")

#need enough samples in the smaller class for the nested 5 fold cv
if np.min(np.bincount(y)) < 10:
    raise ValueError("The smallest class has fewer than 10 samples.")


class MLP(nn.Module):

    def __init__(self, input_size, hidden_layers):
        super().__init__()

        layers = []
        previous = input_size

        for neurons in hidden_layers:
            layers.append(nn.Linear(previous, neurons))
            layers.append(nn.ReLU())
            previous = neurons

        #two output classes, 0 and 1
        layers.append(nn.Linear(previous, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#random search settings
node_options = [16, 32, 64, 128, 256]
batch_size_options = [8, 16, 32]

max_layers = 3

#learning rate and weight decay are sampled on a log scale
lr_range = (1e-4, 1e-2)
wd_range = (1e-5, 1e-2)

max_epochs = 100
min_epochs = 5
patience = 10

#how many random configurations to try in each search
n_random_trials = 30


def log_uniform(rng, low, high):
    value = 10 ** rng.uniform(np.log10(low), np.log10(high))
    #keep 3 significant digits so it prints nicely
    return float(f"{value:.3g}")


def sample_params(rng):
    #randomly pick a full set of hyperparameters

    n_layers = rng.randint(1, max_layers)

    #widths get smaller as the network gets deeper
    hidden_layers = sorted(rng.choices(node_options, k=n_layers), reverse=True)

    return {
        "hidden_layers": hidden_layers,
        "learning_rate": log_uniform(rng, *lr_range),
        "weight_decay": log_uniform(rng, *wd_range),
        "batch_size": rng.choice(batch_size_options)
    }


print("Random trials per search:", n_random_trials)


def train_model(X_train, y_train, X_val, y_val, hidden_layers,
                learning_rate, weight_decay, batch_size, seed):
    #trains with early stopping on the val set, returns val auc and best epoch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    #scaler only sees the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    model = MLP(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = min_epochs
    counter = 0

    for epoch in range(max_epochs):

        model.train()

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        #validation loss
        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()

        current_epoch = epoch + 1

        #no early stopping before min_epochs
        if current_epoch < min_epochs:
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = current_epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    #go back to the best epoch
    model.load_state_dict(best_weights)

    model.eval()

    with torch.no_grad():
        outputs = model(X_val_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    auc = roc_auc_score(y_val, probabilities)

    return auc, best_epoch


#outer 5 fold cv estimates performance
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

#inner 5 fold cv picks hyperparameters
inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

outer_auc_scores = []
outer_results = []
best_params_per_outer_fold = []


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

        hidden_layers = params["hidden_layers"]
        learning_rate = params["learning_rate"]
        weight_decay = params["weight_decay"]
        batch_size = params["batch_size"]

        print(f"\nTrial {trial_number}/{n_random_trials}")
        print(
            "Testing:",
            f"layers={hidden_layers},",
            f"lr={learning_rate},",
            f"weight_decay={weight_decay},",
            f"batch_size={batch_size}"
        )

        inner_auc_scores = []
        inner_best_epochs = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_skf.split(X_outer_train, y_outer_train),
                start=1
        ):

            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]

            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            auc, best_epoch = train_model(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                hidden_layers,
                learning_rate,
                weight_decay,
                batch_size,
                seed=random_state + outer_fold * 100 + trial_number * 10 + inner_fold
            )

            inner_auc_scores.append(auc)
            inner_best_epochs.append(best_epoch)

            print(
                f"  Inner fold {inner_fold} AUC: {auc:.4f} "
                f"| Best epoch: {best_epoch}"
            )

        mean_inner_auc = float(np.mean(inner_auc_scores))
        std_inner_auc = float(np.std(inner_auc_scores, ddof=1))
        median_best_epoch = int(np.median(inner_best_epochs))

        inner_results.append({
            "hidden_layers": hidden_layers,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "mean_inner_auc": mean_inner_auc,
            "std_inner_auc": std_inner_auc,
            "median_best_epoch": median_best_epoch
        })

        print(f"  Mean inner AUC: {mean_inner_auc:.4f} +/- {std_inner_auc:.4f}")
        print(f"  Median best epoch: {median_best_epoch}")

    #best configuration first
    inner_results.sort(key=lambda r: r["mean_inner_auc"], reverse=True)

    pd.DataFrame(inner_results).to_csv(
        f"MLP_OuterFold_{outer_fold}_RandomSearch_Results.csv",
        index=False
    )

    #best hyperparameters, chosen with inner cv only
    best = inner_results[0]

    best_params = {
        "hidden_layers": best["hidden_layers"],
        "learning_rate": best["learning_rate"],
        "weight_decay": best["weight_decay"],
        "batch_size": best["batch_size"],
        "epochs": max(1, best["median_best_epoch"])
    }

    best_inner_auc = best["mean_inner_auc"]

    best_params_per_outer_fold.append({
        "outer_fold": outer_fold,
        **best_params,
        "inner_cv_auc": best_inner_auc
    })

    print("\nBest parameters:")
    print(best_params)
    print(f"Best inner CV AUC: {best_inner_auc:.4f}")

    #scale using the outer training data only
    outer_scaler = StandardScaler()
    X_outer_train_scaled = outer_scaler.fit_transform(X_outer_train)
    X_outer_val_scaled = outer_scaler.transform(X_outer_val)

    X_outer_train_tensor = torch.tensor(X_outer_train_scaled, dtype=torch.float32)
    X_outer_val_tensor = torch.tensor(X_outer_val_scaled, dtype=torch.float32)

    y_outer_train_tensor = torch.tensor(y_outer_train, dtype=torch.long)

    outer_train_loader = DataLoader(
        TensorDataset(X_outer_train_tensor, y_outer_train_tensor),
        batch_size=best_params["batch_size"],
        shuffle=True
    )

    #train a new model on the whole outer training set
    final_outer_model = MLP(
        input_size=X.shape[1],
        hidden_layers=best_params["hidden_layers"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        final_outer_model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )

    outer_seed = random_state + outer_fold * 1000

    torch.manual_seed(outer_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(outer_seed)

    #train for the number of epochs the inner cv picked
    #the outer validation set is not touched here
    for epoch in range(best_params["epochs"]):

        final_outer_model.train()

        for batch_X, batch_y in outer_train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(final_outer_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    #evaluate once on the outer validation set
    final_outer_model.eval()

    with torch.no_grad():
        outputs = final_outer_model(X_outer_val_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    outer_auc = roc_auc_score(y_outer_val, probabilities)

    outer_auc_scores.append(outer_auc)

    outer_results.append({
        "outer_fold": outer_fold,
        "outer_auc": outer_auc,
        "inner_cv_auc": best_inner_auc,
        **best_params
    })

    print(f"\nOuter fold {outer_fold} AUC: {outer_auc:.4f}")


#save outer fold results
pd.DataFrame(outer_results).to_csv(
    "MLP_NestedCV_Outer_Results.csv",
    index=False
)

pd.DataFrame(best_params_per_outer_fold).to_csv(
    "MLP_NestedCV_Selected_Parameters.csv",
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

final_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

final_rng = random.Random(random_state + 100)

final_search_results = []

for trial_number in range(1, n_random_trials + 1):

    params = sample_params(final_rng)

    print(f"\nFinal trial {trial_number}/{n_random_trials}")
    print(params)

    fold_auc_scores = []
    fold_best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(final_skf.split(X, y), start=1):

        auc, best_epoch = train_model(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            params["hidden_layers"],
            params["learning_rate"],
            params["weight_decay"],
            params["batch_size"],
            seed=random_state + 5000 + trial_number * 10 + fold
        )

        fold_auc_scores.append(auc)
        fold_best_epochs.append(best_epoch)

    mean_auc = float(np.mean(fold_auc_scores))
    std_auc = float(np.std(fold_auc_scores, ddof=1))
    median_best_epoch = int(np.median(fold_best_epochs))

    final_search_results.append({
        "hidden_layers": params["hidden_layers"],
        "learning_rate": params["learning_rate"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "median_best_epoch": median_best_epoch
    })

    print(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")


final_search_results.sort(key=lambda r: r["mean_auc"], reverse=True)

pd.DataFrame(final_search_results).to_csv(
    "MLP_Final_Hyperparameter_Search.csv",
    index=False
)

final_best = final_search_results[0]

final_hidden_layers = final_best["hidden_layers"]
final_learning_rate = final_best["learning_rate"]
final_weight_decay = final_best["weight_decay"]
final_batch_size = final_best["batch_size"]
final_epochs = max(1, final_best["median_best_epoch"])

print("\nFinal hyperparameters:")
print({
    "hidden_layers": final_hidden_layers,
    "learning_rate": final_learning_rate,
    "weight_decay": final_weight_decay,
    "batch_size": final_batch_size,
    "epochs": final_epochs
})


#train the final model on all the data
#this one is for future predictions, the nested cv auc is the performance estimate
print("\n" + "=" * 60)
print("Training final model on all data")
print("=" * 60)

final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

final_loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=final_batch_size,
    shuffle=True
)

final_model = MLP(
    input_size=X.shape[1],
    hidden_layers=final_hidden_layers
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=final_learning_rate,
    weight_decay=final_weight_decay
)

for epoch in range(final_epochs):

    final_model.train()

    epoch_loss = 0.0

    for batch_X, batch_y in final_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        loss = criterion(final_model(batch_X), batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(final_loader)

    print(f"Epoch {epoch + 1:3d}/{final_epochs} - Loss: {epoch_loss:.4f}")


#save model and scaler
final_model_path = "final_MLP_nestedCV_5fold_model.pth"
scaler_path = "final_MLP_nestedCV_5fold_scaler.pkl"

torch.save(
    {
        "model_state_dict": final_model.state_dict(),
        "input_size": X.shape[1],
        "hidden_layers": final_hidden_layers,
        "learning_rate": final_learning_rate,
        "weight_decay": final_weight_decay,
        "batch_size": final_batch_size,
        "epochs": final_epochs,
        "log_transform": log_transform,
        "gene_names": gene_names,
        "nested_cv_mean_auc": nested_mean_auc,
        "nested_cv_std_auc": nested_std_auc
    },
    final_model_path
)

joblib.dump(final_scaler, scaler_path)

print("\n" + "=" * 60)
print("Final model saved:", final_model_path)
print("Scaler saved:", scaler_path)
print(f"Nested CV AUC: {nested_mean_auc:.4f} +/- {nested_std_auc:.4f}")