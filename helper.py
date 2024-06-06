from datetime import datetime
from dateutil.relativedelta import relativedelta
from holidays import Australia
from scipy.ndimage import gaussian_filter1d
from scipy import stats
# from sqlalchemy import create_engine, text
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars  as pl
from pykalman import KalmanFilter
import re


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
    holidays_df = pl.from_pandas(holidays_df)
    df = pl.from_pandas(df)

    holidays_count = (
        holidays_df.groupby("POSTING_DATE")
        .agg(pl.count().alias("Holidays"))
    )

    df = df.join(
        holidays_count,
        on="POSTING_DATE",
        how="left"
    )
    df = df.with_columns(
        pl.col("Holidays").fill_null(0)
    )
    print("Holidays filled")
    df = df.sort("POSTING_DATE")
    print('Getting KS')
    def apply_ks_filter(s: pl.Series) -> pl.Series:
        s_filled = s.fill_null(value=0)
        initial_state = s_filled[0]
        kf = KalmanFilter(transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_mean=initial_state,
                        initial_state_covariance=1,
                        observation_covariance=1,
                        transition_covariance=0.01)
        state_means, _ = kf.smooth(s_filled.to_numpy())
        return pl.Series(state_means.flatten())
    df = df.groupby("CUSTOMER_SHIPTO").apply(
        lambda group: group.with_columns(
            apply_ks_filter(group["CCH"]).alias("KS_filtered")  # Name the result column "KS_filtered"
        )
    )
    df = df.with_columns(
        pl.col('KS_filtered').diff().over('CUSTOMER_SHIPTO').alias('KSD')
    )
    print('Getting cv2')
    def square_std(s: pl.Series) -> pl.Series:
        std = s.std()
        return pl.Series([0 if std is None else std ** 2])
    # Apply the function to a rolling window of data
    df = df.groupby("CUSTOMER_SHIPTO").apply(
        lambda group: group.with_columns(
            pl.col("CCH").rolling_apply(square_std, window_size=12).alias("cv2")
        )
    )
    print('Getting cv2 for sales')
    df = df.groupby("CUSTOMER_SHIPTO").apply(
        lambda group: group.with_columns(
            pl.col("SALE_QTY").rolling_apply(square_std, window_size=12).alias("cv2_sales")
        )
    )
    print('Getting apply_periodicity')
    def apply_periodicity(s: pl.Series) -> pl.Series:
        try:
            # Convert Polars Series to NumPy array, apply filter and convert back to Polars Series
            smoothed_measurements = gaussian_filter1d(s.to_numpy(), sigma=1.5)
            return pl.Series(smoothed_measurements)
        except Exception as e:
            # Log the exception if needed
            print(f"An error occurred: {e}")
            return pl.Series([0]*len(s))

    # Apply the function to each group
    df = df.groupby("CUSTOMER_SHIPTO").apply(
        lambda group: group.with_columns(
            pl.col("CCH").rolling_apply(apply_periodicity, window_size=12).alias("period")
        )
    )
    print('Getting apply_smooth')
    def apply_smooth(s: pl.Series) -> pl.Series:
        return s.rolling_mean(window_size=12, min_periods=1)
    df = df.groupby("CUSTOMER_SHIPTO").apply(
        lambda group: group.with_columns(
            apply_smooth(pl.col("Holidays")).alias("Smoothed_Holidays")
        )
    )
    return df.to_pandas()


def clean_data(df):
    df.fillna(0, inplace=True)
    df["CUSTOMER_SHIPTO"] = df["CUSTOMER_NUMBER"].astype(str)
    df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"]).dt.to_period('M').dt.to_timestamp()

    df = pl.from_pandas(df)
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

    # for col in numeric_col:
    #     df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in numeric_col:
        df = df.with_columns(pl.col(col).cast(pl.Float64))
    # cat_col = ["INDUSTRY", "PLANT", "INDUSTRY_SUBLEVEL"]
    # df_cat = df[["CUSTOMER_SHIPTO", "POSTING_DATE"] + cat_col]
    cat_col = ["INDUSTRY", "PLANT", "INDUSTRY_SUBLEVEL"]
    ##############################
    # df_grouped = (
    #     df.groupby(["CUSTOMER_SHIPTO", "POSTING_DATE"])[numeric_col].sum().reset_index()
    # )
    df_grouped = df.groupby(["CUSTOMER_SHIPTO", "POSTING_DATE"]).agg(
        [pl.sum(col).alias(col) for col in numeric_col]
    )
    df = df.sort("POSTING_DATE")
    # Find global max and min dates
    global_max_date = df['POSTING_DATE'].max()
    global_min_date = df['POSTING_DATE'].min()

    all_dates = pd.date_range(start=global_min_date, end=global_max_date, freq='MS')
    all_months_df = pd.DataFrame(all_dates, columns=["POSTING_DATE"])

    # Convert the all_months_df DataFrame to Polars
    pl_all_months_df = pl.from_pandas(all_months_df)

    # Get the unique 'CUSTOMER_SHIPTO' values as a Polars Series
    unique_ship_to = df_grouped.select("CUSTOMER_SHIPTO").unique()

    # Create all combinations of 'CUSTOMER_SHIPTO' and 'POSTING_DATE' using a cross join
    all_combinations = unique_ship_to.join(pl_all_months_df, how="cross")

    # Left join the grouped data with all_combinations
    df_full = all_combinations.join(df_grouped, on=["CUSTOMER_SHIPTO", "POSTING_DATE"], how="left")

    # Fill all na columns with 0
    df_full = df_full.fill_null(0)
    df_full = df_full.to_pandas()
    df_full = create_holidays(df_full)
    return df_full

def pl_groupby(df, group_cols, target_col, new_col, window_size):
    # Convert the pandas DataFrame to a polars DataFrame
    df = pl.from_pandas(df)
    df = df.sort(group_cols + [target_col])
    rolling_mean_expr = pl.col(target_col).rolling_mean(window_size).over(group_cols).alias(new_col)
    # Use the lazy API to add the rolling mean as a new column
    df = df.lazy().with_columns(rolling_mean_expr).collect()
    return df.to_pandas()


def create_lags(df):
    df.fillna(0, inplace=True)
    col_to_lag = [
        "CCH",
        "RENTAL_SALES",
        "POSTING_PERIOD",
        "SALE_QTY",
        "CNTD_RENTAL_POSTING_DATE",
        "DAILY_RENT",
        "CNT_POSTING_DATE",
        "PROD_LRGLG_SALE",
        'Holidays',
        "Smoothed_Holidays",
        "KSD"
    ]
    # sort by POSTING DATE
    df = df.sort_values(["CUSTOMER_SHIPTO", "POSTING_DATE"])
    for j in tqdm(col_to_lag):
        for i in range(6):
            df[f"{j}_lag_{i+1}"] = df[f"{j}"].shift(1 + i)

    df = pl_groupby(df, ["CUSTOMER_SHIPTO"], "CCH", "ROLL_MEAN_3", 3)
    df = pl_groupby(df, ["CUSTOMER_SHIPTO"], "CCH", "ROLL_MEAN_6", 6)
    df = pl.from_pandas(df)
    for i in range(6):
        df = df.with_columns(pl.col("CCH").shift(-1 - i).alias(f"CCH_shift_{i+1}"))
    for i in range(6):
        df = df.with_columns(pl.col("Smoothed_Holidays").shift(-1 - i).alias(f"Smoothed_Holidays_shift_{i+1}"))
    return df.to_pandas()

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
    percent_decline = slope / max_cch

    drop = row["CCH_lag_1"] - row["CCH"]

    return slope, percent_decline, max_cch, p_value, period_to_cross_zero, drop

