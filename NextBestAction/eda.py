from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sqlalchemy import create_engine, text
import dill
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_pickle('df_2.pkl')
"""
## Main Data Exploration
"""
df.describe()
df.info()
"""
creating a dummy value for if a customer will purchase additional different products
"""
df['purchase'] = df['NEW_MATNR'].apply(lambda x: 1 if x > 0 else 0)
df['purchase'].hist()

# plot violin correlation with purchase

variables = ['NEW_MATNR', 'CCH','PRODUCT_SALES','CNTD_DANGEROUS','T_1_ENTRIES']
for var in variables:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='purchase', y=var, data=df)
    plt.title(f'Violin plot of {var} vs purchase')
    plt.show()

# do a logistic regression for purchase
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

variables = df.columns
variables = variables.drop(['purchase','NEW_MATNR','NEW_MATKL', 'NEW_ENTRIES','CUSTOMER_NUMBER'])
selected_feat = ['CCH','AVG_CCH','PRODUCT_SALES','CNTD_DANGEROUS','CNTD_SEA','CNTD_WATERWAY','CNTD_DEBIT_ITEMS','CNTD_MATNR','CNTD_MATKL','CNTD_STANDARD','T_1_ENTRIES','T_2_ENTRIES','T_3_ENTRIES']


X = df[variables]
X = df[selected_feat]
y = df['purchase']
# fill na and inf, -inf with 0
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

# show the coefficients with p values
import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# plot the residuals
plt.scatter(est2.fittedvalues, est2.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')

# plot the qq plot, using seaborn
import statsmodels.api as sm
import seaborn as sns

sm.qqplot(est2.resid, line='45')
plt.title('QQ plot of residuals')
plt.show()

#plot the residuals norm
sns.distplot(est2.resid)
plt.title('Residuals distribution')

# plot the feature impt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]
print(selected_feat)



# show the feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
perm_importance = permutation_importance(rf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.title('Feature importance')
plt.show()

# plot the feature importance using shap
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# plot the feature density
for var in selected_feat:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df.loc[df['purchase'] == 0, var], label='Not purchased', shade=True)
    sns.kdeplot(df.loc[df['purchase'] == 1, var], label='Purchased', shade=True)
    plt.title(f'Density plot of {var} vs purchase')
    plt.show()



