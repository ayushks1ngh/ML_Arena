"""Dataset loaders. They return RAW frames; all fitting happens later, on train only."""
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

from .config import DATA_RAW


def load_titanic():
    """Kaggle train.csv (891 rows); falls back to OpenML if the file is missing."""
    path = DATA_RAW / "titanic" / "train.csv"
    if path.exists():
        df = pd.read_csv(path)
        y = df["Survived"].astype(int)
    else:  # fallback (see docs/DATASETS.md)
        df = fetch_openml("titanic", version=1, as_frame=True).frame
        y = df["survived"].astype(int)
        df = df.rename(columns={"pclass": "Pclass", "sex": "Sex", "age": "Age", "sibsp": "SibSp",
                                "parch": "Parch", "fare": "Fare", "embarked": "Embarked", "name": "Name"})
    X = pd.DataFrame({
        "Pclass": df["Pclass"].astype(int),
        "Sex": df["Sex"].astype(object),
        "Age": df["Age"].astype(float),
        "SibSp": df["SibSp"].astype(int),
        "Parch": df["Parch"].astype(int),
        "Fare": df["Fare"].astype(float),
        "Embarked": df["Embarked"].astype(object),
    })
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    title = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    common = {"Mr", "Mrs", "Miss", "Master"}
    X["Title"] = title.where(title.isin(common), "Rare").astype(object)
    return X, y.rename("Survived")


def load_california():
    """scikit-learn copy of California Housing: 20,640 rows, 8 numeric features."""
    d = fetch_california_housing(as_frame=True)
    return d.data, d.target.rename("MedHouseVal")


def load_mall():
    """Mall Customers: 200 rows. Gender kept in the frame; clustering code decides what to use."""
    return pd.read_csv(DATA_RAW / "mall" / "Mall_Customers.csv")
