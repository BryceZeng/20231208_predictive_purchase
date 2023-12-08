from datetime import datetime
from dateutil.relativedelta import relativedelta
from holidays import Australia
from scipy.ndimage import gaussian_filter1d
# from sqlalchemy import create_engine, text
from tqdm import tqdm
import numpy as np
import pandas as pd
import re


# def import_data(df, sql, min_date):
#     """Import in the sql query and set the min_date (optional)
#     default to 24 months"""
#     if not min_date:
#         current_date = datetime.today() - relativedelta(days=10)
#         min_date = (current_date - relativedelta(months=24)).strftime("%Y-%m-01")

#     with open("./sql/raw_data.sql", "r") as file:
#         sql = file.read()
#     sql = re.sub(r"min_date_set", min_date, sql)

#     connection_string = (
#         "mssql+pyodbc://Customers_at_Risk_PRD:sTzmxZ3oLKRt64QoJ@mlggae00sql005.linde.lds.grp,1433/APAC_DATA_REPO?"
#         "driver=ODBC+Driver+17+for+SQL+Server&"
#         "Encrypt=no&"
#         "TrustServerCertificate=no&"
#         "Connection+Timeout=300&"
#         "ColumnEncryption=Enabled&"
#         "DisableCertificateVerification=yes"
#     )
#     engine = create_engine(connection_string)
#     with engine.connect() as conn:
#         df = pd.read_sql_query(text(sql), conn)
#     return df


def create_holidays(df):
    years = df["POSTING_DATE"].dt.year.unique()
    au_holidays = Australia(years=years.tolist())
    # Create a DataFrame from the Australia holidays
    holidays_df = pd.DataFrame.from_dict(au_holidays, orient="index").reset_index()
    holidays_df.columns = ["Date", "Holiday"]
    holidays_df["Date"] = pd.to_datetime(holidays_df["Date"])

    # Extract month and year from dates and group by them to count holidays
    holidays_df["POSTING_DATE"] = (
        holidays_df["Date"].dt.to_period("M").dt.to_timestamp()
    )
    holidays_count = (
        holidays_df.groupby(["POSTING_DATE"]).size().reset_index(name="Holidays")
    )

    df = pd.merge(df, holidays_count, how="left", on=["POSTING_DATE"])
    df["Holidays"] = df["Holidays"].fillna(0)

    # Create smooth holidays
    sigma = 1.5  # This is the standard deviation for the Gaussian kernel.

    def smooth(group):
        return gaussian_filter1d(group, sigma)

    df["Smoothed_Holidays"] = df.groupby("CUSTOMER_SHIPTO")["Holidays"].transform(
        smooth
    )

    # Assign sqr standard variation of CCH to each row
    def squared_std(x):
        return np.std(x) ** 2

    grouped = df.groupby("CUSTOMER_SHIPTO").expanding()
    cv2 = grouped["CCH"].apply(squared_std, raw=True).reset_index(level=0, drop=True)
    df = df.assign(cv2=cv2)

    return df


def clean_data(df):
    df.fillna(0, inplace=True)
    # df["CUSTOMER_SHIPTO"] = df.apply(
    #     lambda row: f"{row['CUSTOMER_NUMBER']}_{row['SHIP_TO']}", axis=1
    # )
    df["CUSTOMER_SHIPTO"] = df["CUSTOMER_NUMBER"].astype(str)
    df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
    numeric_col = [
        "CNTD_RENTAL_POSTING_DATE",
        "CNTD_POSTING_DATE",
        "CNT_POSTING_DATE",
        "AVG_DOCUMENT_ISSUE_DIFF",
        "AVG_POST_ISSUE_DIFF",
        "REFERENCE_ITEMS",
        "POSTING_PERIOD",
        "ORDER_NO",
        "CCH",
        "DAY_BETWEEN_POSTING",
        "SALE_QTY",
        "RENTAL_BILLED_QTY",
        "PRODUCT_SALES",
        "RENTAL_SALES",
        "DELIVERY",
        "DAILY_RENT",
        "MONTHLY_RENT",
        "QUARTERLY_RENT",
        "ANNUAL_RENT",
        "Other_Rent_Period",
        "DISCOUNT_RATIO",
        "MATERIAL_020112_SALE",
        "MATERIAL_050299_SALE",
        "MATERIAL_020110_SALE",
        "MATERIAL_020104_SALE",
        "MATERIAL_111899_SALE",
        "MATERIAL_051299_SALE",
        "PROD_SMLLD2_SALE",
        "PROD_1MEDLE_SALE",
        "PROD_SMLLD_SALE",
        "PROD_5MEDLE_SALE",
        "PROD_LRGLG_SALE",
        "PROD_4MEDLE2_SALE",
        "PROD_8LRGLG_SALE",
    ]

    for col in numeric_col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_col = ["INDUSTRY", "PLANT", "INDUSTRY_SUBLEVEL"]
    df_cat = df[["CUSTOMER_SHIPTO", "POSTING_DATE"] + cat_col]

    ##############################
    df_grouped = (
        df.groupby(["CUSTOMER_SHIPTO", "POSTING_DATE"])[numeric_col].sum().reset_index()
    )

    global_max_date = df["POSTING_DATE"].max()
    global_min_date = df["POSTING_DATE"].min()

    # Create a new dataframe with a multi-index of 'SHIP_TO' and all dates from min to max for each 'SHIP_TO'
    idx = pd.MultiIndex.from_product(
        [
            df_grouped["CUSTOMER_SHIPTO"].unique(),
            pd.date_range(start=global_min_date, end=global_max_date, freq="MS"),
        ],
        names=["CUSTOMER_SHIPTO", "POSTING_DATE"],
    )
    # Reindex the original DataFrame with the new MultiIndex
    df_full = df_grouped.set_index(["CUSTOMER_SHIPTO", "POSTING_DATE"]).reindex(idx)
    df_full = df_full.reset_index()
    df_full.fillna(0, inplace=True)

    ##############################
    df_full = pd.merge(df_full, df_cat)
    df_full["INDUSTRY"] = df_full.groupby("CUSTOMER_SHIPTO")["INDUSTRY"].transform(
        lambda x: x.ffill().bfill()
    )
    df_full["PLANT"] = df_full.groupby("CUSTOMER_SHIPTO")["PLANT"].transform(
        lambda x: x.ffill().bfill()
    )
    df_full["INDUSTRY_SUBLEVEL"] = df_full.groupby("CUSTOMER_SHIPTO")[
        "INDUSTRY_SUBLEVEL"
    ].transform(lambda x: x.ffill().bfill())
    df_full.fillna(0, inplace=True)

    #####################
    # Add Holidays
    df_full = create_holidays(df_full)

    return df_full


def create_lags(df):
    col_to_lag = [
        "CCH",
        "RENTAL_SALES",
        "POSTING_PERIOD",
        "SALE_QTY",
        "CNTD_RENTAL_POSTING_DATE",
        "DAILY_RENT",
        "CNT_POSTING_DATE",
        "PROD_LRGLG_SALE",
        "Smoothed_Holidays",
    ]
    for j in tqdm(col_to_lag):
        for i in range(3):
            df[f"{j}_lag_{i+1}"] = df[f"{j}"].shift(1 + i)

    df["ROLL_MEAN_3"] = df.groupby("CUSTOMER_SHIPTO")["CCH"].transform(
        lambda x: x.rolling(3, min_periods=0).mean()
    )
    df["ROLL_MEAN_6"] = df.groupby("CUSTOMER_SHIPTO")["CCH"].transform(
        lambda x: x.rolling(6, min_periods=0).mean()
    )
    for i in range(6):
        df[f"CCH_shift_{i+1}"] = df["CCH"].shift(-1 - i)
    for i in range(6):
        df[f"Smoothed_Holidays_shift_{i+1}"] = df["Smoothed_Holidays"].shift(-1 - i)
    return df
