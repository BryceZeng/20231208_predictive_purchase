from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import dill
import numpy as np
import pandas as pd
import lightgbm as lgb


def train_classifer(df):
    model = None
    return model


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
