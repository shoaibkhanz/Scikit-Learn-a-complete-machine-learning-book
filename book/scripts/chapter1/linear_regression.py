# %% [markdown]
# # Linear Regression


# %%
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# %%
# load the data from scikit-learn (a bunch is returned)
boston = load_boston()
# required components
data = boston["data"]
target = boston["target"]
cols = boston["feature_names"]


# %%
def get_var(df, var_name):
    globals()[var_name] = df


# fmt:on
(
    pd.DataFrame(data, columns=cols)
    .assign(target=target)
    .dropna(axis=1)
    .rename(str.lower, axis=1)
    .pipe(get_var, "df1")
)

Y = df1["target"]
X = df1.drop(columns="target", axis=1)


# %%

print(X.shape)
print(Y.shape)

#%%
(X_train, Y_train, X_test, Y_test,) = train_test_split(
    X, Y, test_size=0.3, random_state=42, shuffle=False,
)

#%%
print(X_train.shape)
print(Y_test.shape)

# %%
