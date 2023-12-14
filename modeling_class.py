from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import dill
import numpy as np
import pandas as pd
import lightgbm as lgb
import helper
from sklearn.model_selection import train_test_split
import lightgbm as lgb

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
    len(df)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.25)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
        params = {"boosting_type": "dart", "objective": "mse"}
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=lgb_test,
            # early_stopping_rounds=150,
        )
        return gbm
    y1_pred = prediction_lgb(X, y1)
    y2_pred = prediction_lgb(X, y2)
    y3_pred = prediction_lgb(X, y3)
    y4_pred = prediction_lgb(X, y4)
    y5_pred = prediction_lgb(X, y5)
    y6_pred = prediction_lgb(X, y6)

    model_list = [y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred]
    # with open("model_list3.pkl", "wb") as f:
    #     dill.dump(model_list, f)
    return model_list


def predict_classifer(df_long, df_wide, model, start_date="2022-06-01"):
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
    df_out = df_wide[
        ["CUSTOMER_SHIPTO", "POSTING_DATE", "CCH", "CCH_lag_1", "CCH_lag_2"]
    ]
    df_out["pred1"] = model[0].predict(X)
    df_out["pred2"] = model[1].predict(X)
    df_out["pred3"] = model[2].predict(X)
    df_out["pred4"] = model[3].predict(X)
    df_out["pred5"] = model[4].predict(X)
    df_out["pred6"] = model[5].predict(X)
    return df_out

def explainer_d():
    import dalex as dx
    return explainer_m