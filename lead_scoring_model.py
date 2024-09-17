#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
import random
seed = 42
random.seed(seed)


# In[2]:


df = pl.read_parquet("leads.parquet")


# In[3]:


df.describe()


# In[4]:


df.null_count().sum() / df.shape[0]


# In[5]:


# Listing out the columns that having null value
df.null_count().sum().columns


# In[6]:


df.shape[0]


# ### Columns selection

# In[7]:


df_pandas = df.to_pandas()
all_cats_columns = df.select(cs.string()).columns
binary_cols = []
cate_cols = []
for c in all_cats_columns:
    if len(df_pandas[c].unique().tolist()) == 2:
        binary_cols.append(c)
    else:
        cate_cols.append(c)
print(f"binary_cols ({len(binary_cols)}): {binary_cols}")
print(f"cate_cols ({len(cate_cols)}): {cate_cols}")
target = "Converted"
num_columns = df.select(cs.integer() | cs.float()).columns
print(f"num_columns ({len(num_columns)}): {num_columns}")
print(f"Target columns: {target}")
print(f"Total columns: {len(df.columns)}")


# In[8]:


all_cats_columns


# In[9]:


len(all_cats_columns)


# ## Visualising Numeric Variables
# 
# Let's make a pairplot of all the numeric variables

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

df_viz = df.to_pandas()


# In[11]:


sns.pairplot(df_viz[num_columns])
plt.show()


# ## Visualising Categorical Variables
# 
# As you might have noticed, there are a few categorical variables as well. Let's make a boxplot for some of these variables.

# ## Cat Plot

# In[12]:


import seaborn as sns

# this prove the point that most of leads dont want to called.
sns.catplot(data=df_viz, x="Do Not Call", y="Converted", kind="box")


# In[13]:


# As tags are the most importance feature in prediction. we should consider the category with the most converted time counted.
df_viz.groupby("Tags")["Converted"].apply(
    lambda x: x.value_counts().sort_values(ascending=False)
)


# In[14]:


# We analyze other characteristics show that people want to make call and converted also.
df.filter((pl.col("Do Not Call") == "Yes") & (pl.col("Converted") == 1))


# In[15]:


from scipy.stats import chi2_contingency
import math


def cramers_v(x, y):
    # Create a contingency table
    contingency_table = pd.crosstab(x, y)

    # Calculate Chi-Square statistic
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim))
    if math.isnan(v):
        return 0
    else:
        return v


# In[16]:


# Show the correlation between categorical columns and target using Chi-Square statistic
corr_list = [{c: cramers_v(df_viz[c], df_viz[target])} for c in all_cats_columns]
sorted_corr = sorted(corr_list, key=lambda x: list(x.values())[0], reverse=True)
new_cats_columns = [list(c.keys())[0] for c in sorted_corr if list(c.values())[0] > 0]


# In[17]:


sorted_corr


# In[18]:


new_cats_columns


# In[19]:


# filter out columns that having zero-related to the target (0 or 0.0)
binary_cols = [c for c in binary_cols if c in new_cats_columns]
cate_cols = [c for c in cate_cols if c in new_cats_columns]
print(f"binary_cols: {binary_cols}")
print(f"cate_cols: {cate_cols}")


# In[20]:


# Spearman correlation between features and target
spearman_corr = df_viz[num_columns].corr()
print("\nSpearman Correlation:")
print(spearman_corr[target])


# ## Data Preprocessing

# In[21]:


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    LabelBinarizer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import train_test_split


# In[22]:


# remove target since it will be processed in an other part
num_columns.remove(target)
cate_cols.remove("Prospect ID")

y = df_pandas[target]


# ### Preprocessing X_train

# In[23]:


# handle nominal variables
categorical_pipeline = Pipeline(
    steps=[
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("target_encoder", TargetEncoder(smooth="auto")),
    ]
)

# Handle binary variables
binary_pipeline = Pipeline(
    steps=[
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal_encoder", OrdinalEncoder()),
    ]
)

# Handle numerical variables
numerical_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
    ]
)

# Combine the pipelines into a ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, num_columns),
        ("cat", categorical_pipeline, cate_cols),
        ("binary", binary_pipeline, binary_cols),
    ],
)

# Transform all values and standardize the values
data_preprocessing_pipeline = Pipeline(
    steps=[
        ("column_transformer", column_transformer),
        # In a typical scikit-learn pipeline, the StandardScaler (or any scaling/normalization step) is usually placed after all other preprocessing steps but before any model fitting. The purpose of scaling is to standardize the range of features so that they contribute equally to the model's performance.
        ("standardize", StandardScaler()),
    ]
)


# In[24]:


X = data_preprocessing_pipeline.fit_transform(df_pandas.drop(columns=target, axis=1), y)
X = pd.DataFrame(X, columns=data_preprocessing_pipeline.get_feature_names_out())
X.head()


# ## Split data

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y
)


# ## Perform tuning hyper params with CV

# In[26]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
# Define the hyperparameter search space
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
    "penalty": ["l1", "l2"],  # Type of regularization
    "multi_class": ["auto", "ovr", "multinomial"],
    "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
}
log_reg = LogisticRegression()

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring="accuracy", verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")


# ## Model Evaluation

# In[27]:


# Evaluate the best model on the test set
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

f1_scores = f1_score(y_test, y_pred)
roc_auc_scores = roc_auc_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc_scores:.4f}")
print(f"F1 score: {f1_scores:.4f}")


# ## Train Random forest to retrive the feature importance

# In[28]:


from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importances = rf_model.feature_importances_

# Combine with feature names
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})

# Sort features by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print(importance_df)

