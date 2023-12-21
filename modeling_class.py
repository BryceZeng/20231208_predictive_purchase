from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import dill
import numpy as np
import pandas as pd
import lightgbm as lgb
import helper
from sklearn.model_selection import train_test_split
import dalex as dx
from scipy import stats

variables = [
        "period_1_sample_mean",
        "period_2_sample_mean",
        "period_3_sample_mean",
        "period_4_sample_mean",
        "period_5_sample_mean",
        "period_6_sample_mean",
        "period_1_uncertainty",
        "period_2_uncertainty",
        "period_3_uncertainty",
        "period_4_uncertainty",
        "period_5_uncertainty",
        "period_6_uncertainty",
        "CCH_lag_1",
        "CCH_lag_2",
        "CCH_lag_3",
        "CNTD_RENTAL_POSTING_DATE",
        "REFERENCE_ITEMS",
        "SALE_QTY",
        "CNT_POSTING_DATE",
        "PRODUCT_SALES",
        "RENTAL_BILLED_QTY",
        "RENTAL_SALES",
        "DAILY_RENT",
        "Smoothed_Holidays",
        "Smoothed_Holidays_lag_1",
        "Smoothed_Holidays_shift_1",
        "Smoothed_Holidays_shift_2",
        "Smoothed_Holidays_shift_3",
        "Smoothed_Holidays_shift_4",
        "Smoothed_Holidays_shift_5",
        "Smoothed_Holidays_shift_6",
        "cv2",
        "POSTING_PERIOD",
        "PROD_LRGLG_SALE",
        "PROD_5MEDLE_SALE",
        "RENTAL_SALES_lag_1",
        "RENTAL_SALES_lag_2",
        "RENTAL_SALES_lag_3",
        "SALE_QTY_lag_1",
        "SALE_QTY_lag_2",
        "SALE_QTY_lag_3",
        "DAILY_RENT_lag_1",
        "DAILY_RENT_lag_2",
        "DAILY_RENT_lag_3",
        "CNT_POSTING_DATE_lag_1",
        "CNT_POSTING_DATE_lag_2",
        "CNT_POSTING_DATE_lag_3",
        "PROD_LRGLG_SALE_lag_1",
        "PROD_LRGLG_SALE_lag_2",
        "PROD_LRGLG_SALE_lag_3",
        "ROLL_MEAN_3",
        "ROLL_MEAN_6",
    ]
def train_classifer(df):
    df = pd.read_csv("data/data_8.csv")
    df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
    # df.head()
    print(len(df))
    print(df["POSTING_DATE"].max())
    df = df[df["POSTING_DATE"]<='2023-11-01']
    df = df[df["POSTING_DATE"]>='2022-01-01']
    df = helper.clean_data(df)
    df = helper.create_lags(df)
    df_reg = pd.read_csv("data/for_regression3.csv")
    columns = [('period_1_lower_bound', 'period_2_lower_bound','period_3_lower_bound','period_4_lower_bound','period_5_lower_bound','period_6_lower_bound','period_1_sample_mean','period_2_sample_mean','period_3_sample_mean','period_4_sample_mean','period_5_sample_mean','period_6_sample_mean',
            'period_1_upper_bound','period_2_upper_bound','period_3_upper_bound','period_4_upper_bound','period_5_upper_bound','period_6_upper_bound','POSTING_DATE'
            )] +[('period_1_lower_bound.'+str(i), 'period_2_lower_bound.'+str(i),'period_3_lower_bound.'+str(i),'period_4_lower_bound.'+str(i),'period_5_lower_bound.'+str(i),'period_6_lower_bound.'+str(i),'period_1_sample_mean.'+str(i),'period_2_sample_mean.'+str(i),'period_3_sample_mean.'+str(i),'period_4_sample_mean.'+str(i),'period_5_sample_mean.'+str(i),'period_6_sample_mean.'+str(i),
            'period_1_upper_bound.'+str(i),'period_2_upper_bound.'+str(i),'period_3_upper_bound.'+str(i),'period_4_upper_bound.'+str(i),'period_5_upper_bound.'+str(i),'period_6_upper_bound.'+str(i),'POSTING_DATE.'+str(i)) for i in range(1, 14)]

    dfs = pd.DataFrame()
    for i in tqdm(columns):
        df_melted = df_reg.melt(id_vars='CUSTOMER_SHIPTO', value_vars=list(i), var_name='variable', value_name='value')
        df_melted = df_melted.pivot(index='CUSTOMER_SHIPTO', columns='variable', values='value').reset_index()
        df_melted = df_melted.rename(columns=lambda x: x.split('.')[0])
        dfs = pd.concat([dfs, df_melted], ignore_index=True, axis=0)
    dfs["POSTING_DATE"] = pd.to_datetime(dfs["POSTING_DATE"])
    df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
    df_wide = dfs.merge(df, on=["CUSTOMER_SHIPTO", "POSTING_DATE"])
    ### Combining the fields()
    for period in tqdm(range(1, 7)):
        column_name = f"period_{period}_uncertainty"
        upper_bound = df_wide[f"period_{period}_upper_bound"]
        lower_bound = df_wide[f"period_{period}_lower_bound"]
        df_wide[column_name] = upper_bound - lower_bound
    df_wide = df_wide[df_wide['POSTING_DATE'] <='2023-05-01']
    df_wide = df_wide.dropna()
    df_wide = pd.read_pickle('df_wide.pkl')


    # mask = (df_wide[['CCH_shift_1','CCH_shift_2','CCH_shift_3','CCH_shift_4','CCH_shift_5','CCH_shift_6']] <= 0)
    # row_mask = mask.all(axis=1)
    # df_wide = df_wide[~row_mask]
    # df_wide = df_wide.reset_index()

    for column in variables:
        df_wide[column] = df_wide[column].astype(float)
    X = df_wide[variables]
    y1 = df_wide["CCH_shift_1"]
    y2 = df_wide["CCH_shift_2"]
    y3 = df_wide["CCH_shift_3"]
    y4 = df_wide["CCH_shift_4"]
    y5 = df_wide["CCH_shift_5"]
    y6 = df_wide["CCH_shift_6"]

    def prediction_lgb(X, y1):
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.30)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # params = {"boosting_type": "dart", "objective": "mse","learning_rate": 0.02,"max_depth": 3,"num_leaves": 12}
        params = {"boosting_type": "gbdt",
                "objective": "regression",
                'metric':['l1','l2'],
                "tree":"voting",
                "learning_rate": 0.01,
                'num_iterations':3000,
                'early_stopping_round':5,
                # "max_depth": 3,
                # "bagging_fraction":0.6,
                "data_sample_strategy":'goss',
                # "num_leaves": 12
                'force_row_wise':True
                }
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=lgb_test,
        )
        return gbm
    y1_pred = prediction_lgb(X, y1)
    y2_pred = prediction_lgb(X, y2)
    y3_pred = prediction_lgb(X, y3)
    y4_pred = prediction_lgb(X, y4)
    y5_pred = prediction_lgb(X, y5)
    y6_pred = prediction_lgb(X, y6)

    model_list = [y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred]
    # with open("model_list3b.pkl", "wb") as f:
    #     dill.dump(model_list, f)
    return model_list

def calculate_slope(row):
    data = [
        row["CCH_lag_2"],
        row["CCH_lag_1"],
        row["CCH"],
        row["pred1"],
        row["pred2"],
        row["pred3"],
        row["pred4"],
        row["pred5"],
        row["pred6"],
    ]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(data)), data
    )
    if slope < 0:
        period_to_cross_zero = abs(intercept / slope)
    else:
        period_to_cross_zero = None
    max_cch = max([row["CCH_lag_2"], row["CCH_lag_1"], row["CCH"]])
    if max_cch != 0:
        percent_decline = slope / max_cch
    else:
        percent_decline = 0

    drop = row["CCH_lag_1"] - row["CCH"]

    return slope, percent_decline, max_cch, p_value, period_to_cross_zero, drop


def predict_classifer(df_long, df_wide, model,gbm_explainer,
                    start_date="2022-06-01",
                    percent_1=0.15,
                    percent_2=0.15,
                    max_cch=5,
                    PRODUCT_SALES=5000):
    tqdm.pandas()
    df_wide["POSTING_DATE"] = df_wide["PREDICTION_DATE"]
    df_wide["POSTING_DATE"] = pd.to_datetime(df_wide["POSTING_DATE"])
    df_long["POSTING_DATE"] = pd.to_datetime(df_long["POSTING_DATE"])
    df_wide = df_wide.merge(df_long, on=["CUSTOMER_SHIPTO", "POSTING_DATE"])
    df_wide = df_wide[df_wide["POSTING_DATE"] == start_date]
    ### Combining the fields()
    for period in tqdm(range(1, 7)):
        column_name = f"period_{period}_uncertainty"
        upper_bound = df_wide[f"period_{period}_upper_bound"]
        lower_bound = df_wide[f"period_{period}_lower_bound"]
        df_wide[column_name] = upper_bound - lower_bound

    X = df_wide[variables]

    df_wide["pred1"] = np.maximum(model[0].predict(X), -5)
    df_wide["pred2"] = np.maximum(model[1].predict(X), -5)
    df_wide["pred3"] = np.maximum(model[2].predict(X), -5)
    df_wide["pred4"] = np.maximum(model[3].predict(X), -5)
    df_wide["pred5"] = np.maximum(model[4].predict(X), -5)
    df_wide["pred6"] = np.maximum(model[5].predict(X), -5)

    df_wide[["slope", "percent_decline", "max_cch", "p_value", "cross_zero", "drop"]] = df_wide.apply(
        calculate_slope, axis=1, result_type="expand"
    )
    # df_out['explainer'] = X.progress_apply(lambda row: explainer_d(gbm_explainer, row), axis=1)
    def conditional_apply(row):
        corresponding_row = df_wide.loc[row.name]
        if (abs(corresponding_row['percent_decline']) >= percent_1 and corresponding_row['PRODUCT_SALES'] >= PRODUCT_SALES) or \
            (abs(corresponding_row['percent_decline']) >= percent_2 and corresponding_row['max_cch'] >= max_cch) or \
            abs(corresponding_row['drop']) >= 3:
            return explainer_d(gbm_explainer, row)
        else:
            return ''
    df_wide['explainer'] = X.progress_apply(conditional_apply, axis=1)
    df_out = df_wide[
        ["CUSTOMER_SHIPTO", "POSTING_DATE", "CCH", "CCH_lag_1", "CCH_lag_2",
        "slope", "percent_decline", "max_cch", "p_value", "cross_zero", "drop",
        "pred1","pred2","pred3","pred4","pred5","pred6",'explainer']
    ]
    return df_out

def build_explainer(y3_pred,X, y3):
    sample_size = 5000
    sample_indices = np.random.choice(X.index, size=sample_size, replace=False)
    X_sample = X.loc[sample_indices]
    y3_sample = y3.loc[sample_indices]
    gbm_explainer = dx.Explainer(y3_pred, X_sample, y3_sample, label="gbm")
    return gbm_explainer

def explainer_d(gbm_explainer, instance):
    prediction_breakdown = gbm_explainer.predict_parts(instance, B=5, N=500, type='break_down').result
    replacements = {
        r'period_\d+_' : '',
        r'_lag_\d' : '',
        r'_shift_\d' : '',
        r'Smoothed_Holidays' : 'Holidays',
        r'cv2' : 'Cyclicality',
        r'_\d' : '',
        r'sample_mean' : 'Trend',
        r'ROLL_MEAN' : 'Past_CCH',
        r'uncertainty' : 'Uncertainty',
        r'REFERENCE_ITEMS' : 'Cnt_Items',
        r'CNTD_RENTAL_POSTING_DATE': 'Rental_Freq',
        r'PROD_LRGLG_SALE' : 'Prod_1LRGLG',
        r'PROD_5MEDLE_SALE' : 'Prod_15MEDLE'
    }
    prediction_breakdown['variable_name'] = prediction_breakdown['variable_name'].replace(replacements, regex=True)
    prediction_breakdown = prediction_breakdown.groupby('variable_name')['contribution'].sum().abs().reset_index()
    prediction_breakdown = prediction_breakdown[~prediction_breakdown['variable_name'].str.contains('^$|intercept|Trend|Uncertainty')]
    top_contributions = prediction_breakdown.nlargest(3, 'contribution')['variable_name'].tolist()
    string = f"This is due to {top_contributions[0]}, {top_contributions[1]}, and {top_contributions[2]}. "

    return string


