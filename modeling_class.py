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
from sklearn.metrics import mean_squared_error

variables = [
        "period_1_sample_mean",
        "period_2_sample_mean",
        "period_3_sample_mean",
        "period_4_sample_mean",
        "period_5_sample_mean",
        "period_6_sample_mean",
        "CNT_POSTING_DATE",
        "CCH",
        "CCH_lag_1",
        "CCH_lag_2",
        "CCH_lag_3",
        "CCH_lag_4",
        "CCH_lag_5",
        "CCH_lag_6",
        "SALE_QTY",
        "DAILY_RENT",
        "MONTHLY_RENT",
        "QUARTERLY_RENT",
        "ANNUAL_RENT",
        "Smoothed_Holidays_shift_1",
        "Smoothed_Holidays_shift_2",
        "Smoothed_Holidays_shift_3",
        "Smoothed_Holidays_shift_4",
        "Smoothed_Holidays_shift_5",
        "Smoothed_Holidays_shift_6",
        "period_1_lower_bound",
        "period_2_lower_bound",
        "period_3_lower_bound",
        "period_4_lower_bound",
        "period_5_lower_bound",
        "period_6_lower_bound",
        "period_1_upper_bound",
        "period_2_upper_bound",
        "period_3_upper_bound",
        "period_4_upper_bound",
        "period_5_upper_bound",
        "period_6_upper_bound",
        "less12",
        "RENTAL_BILLED_QTY",
        "PRODUCT_SALES",
        "RENTAL_SALES",
        "cv2",
        "cv2_sales",
        "RENTAL_SALES_lag_1",
        "RENTAL_SALES_lag_2",
        "RENTAL_SALES_lag_3",
        "RENTAL_SALES_lag_4",
        "RENTAL_SALES_lag_5",
        "RENTAL_SALES_lag_6",
        "SALE_QTY_lag_1",
        "SALE_QTY_lag_2",
        "SALE_QTY_lag_3",
        "SALE_QTY_lag_4",
        "SALE_QTY_lag_5",
        "SALE_QTY_lag_6",
        "DELIVERY",
        "MATERIAL_020112_SALE",
        "MATERIAL_020110_SALE",
        "MATERIAL_020104_SALE",
    ]

# variables = [
#     "period_1_sample_mean",
#     "period_2_sample_mean",
#     "period_3_sample_mean",
#     "period_4_sample_mean",
#     "period_5_sample_mean",
#     "period_6_sample_mean",
#     "CNT_POSTING_DATE",
#     "CCH",
#     "CCH_lag_1",
#     "CCH_lag_2",
#     "CCH_lag_3",
#     "CCH_lag_4",
#     "CCH_lag_5",
#     "CCH_lag_6",
#     "SALE_QTY",
#     "DAILY_RENT",
#     "MONTHLY_RENT",
#     "QUARTERLY_RENT",
#     "ANNUAL_RENT",
#     "Smoothed_Holidays_shift_1",
#     "Smoothed_Holidays_shift_2",
#     "Smoothed_Holidays_shift_3",
#     "Smoothed_Holidays_shift_4",
#     "Smoothed_Holidays_shift_5",
#     "Smoothed_Holidays_shift_6",
#     "period_1_lower_bound",
#     "period_2_lower_bound",
#     "period_3_lower_bound",
#     "period_4_lower_bound",
#     "period_5_lower_bound",
#     "period_6_lower_bound",
#     "period_1_upper_bound",
#     "period_2_upper_bound",
#     "period_3_upper_bound",
#     "period_4_upper_bound",
#     "period_5_upper_bound",
#     "period_6_upper_bound",
#     "less12",
#     "RENTAL_BILLED_QTY",
#     "PRODUCT_SALES",
#     "RENTAL_SALES",
#     "cv2",
#     "cv2_sales",
#     "RENTAL_SALES_lag_1",
#     "RENTAL_SALES_lag_2",
#     "RENTAL_SALES_lag_3",
#     "RENTAL_SALES_lag_4",
#     "RENTAL_SALES_lag_5",
#     "RENTAL_SALES_lag_6",
#     "SALE_QTY_lag_1",
#     "SALE_QTY_lag_2",
#     "SALE_QTY_lag_3",
#     "SALE_QTY_lag_4",
#     "SALE_QTY_lag_5",
#     "SALE_QTY_lag_6",
#     "DELIVERY",
#     "MATERIAL_020112_SALE",
#     "MATERIAL_020110_SALE",
#     "MATERIAL_020104_SALE",
# ]
def train_classifer(df):
    # read in df from pickle
    dfa = pd.read_pickle("df_all.pkl")
    df = pd.read_pickle("df.pkl")
    # subset CUSTOMER_SHIPTO from dfa
    df = df[df["CUSTOMER_SHIPTO"].isin(dfa["CUSTOMER_SHIPTO"])]

    dfa.columns
    dfa = dfa[
        [
            "CUSTOMER_SHIPTO",
            "period_1_lower_bound",
            "period_2_lower_bound",
            "period_3_lower_bound",
            "period_4_lower_bound",
            "period_5_lower_bound",
            "period_6_lower_bound",
            "period_1_sample_mean",
            "period_2_sample_mean",
            "period_3_sample_mean",
            "period_4_sample_mean",
            "period_5_sample_mean",
            "period_6_sample_mean",
            "period_1_upper_bound",
            "period_2_upper_bound",
            "period_3_upper_bound",
            "period_4_upper_bound",
            "period_5_upper_bound",
            "period_6_upper_bound",
            "PREDICTION_DATE",
        ]
    ]
    list_col = dfa.columns.tolist()

    dfa["POSTING_DATE"] = pd.to_datetime(dfa["PREDICTION_DATE"])
    df = df.merge(dfa, on=["CUSTOMER_SHIPTO", "POSTING_DATE"], how="inner")
    list_col = df.columns.tolist()

    print(len(df))
    ### Combining the fields()
    for period in tqdm(range(1, 7)):
        column_name = f"period_{period}_uncertainty"
        upper_bound = df[f"period_{period}_upper_bound"]
        lower_bound = df[f"period_{period}_lower_bound"]
        df[column_name] = upper_bound - lower_bound

    # identify what columns are important for CCH_shift_3 apply a lasso regression
    Y = df["CCH_shift_6"]
    elements_to_remove = [
        "POSTING_DATE",
        "cv2_sales",
        "CCH_shift_1",
        "CCH_shift_2",
        "CCH_shift_3",
        "CCH_shift_4",
        "CCH_shift_5",
        "CCH_shift_6",
        "ORDER_NO",
        "PREDICTION_DATE",
        "CUSTOMER_SHIPTO",
    ]
    for element in elements_to_remove:
        list_col.remove(element)
    X = df[list_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
    import numpy as np
    from sklearn.linear_model import LassoCV

    lasso = LassoCV()
    lasso.fit(X_train, y_train)
    # get the Important features
    features_select = [
        list_col[i] for i in range(len(lasso.coef_)) if lasso.coef_[i] != 0
    ]
    variables = [
        "period_1_sample_mean",
        "period_2_sample_mean",
        "period_3_sample_mean",
        "period_4_sample_mean",
        "period_5_sample_mean",
        "period_6_sample_mean",
        "CNT_POSTING_DATE",
        "CCH",
        "CCH_lag_1",
        "CCH_lag_2",
        "CCH_lag_3",
        "CCH_lag_4",
        "CCH_lag_5",
        "CCH_lag_6",
        "SALE_QTY",
        "DAILY_RENT",
        "MONTHLY_RENT",
        "QUARTERLY_RENT",
        "ANNUAL_RENT",
        "Smoothed_Holidays_shift_1",
        "Smoothed_Holidays_shift_2",
        "Smoothed_Holidays_shift_3",
        "Smoothed_Holidays_shift_4",
        "Smoothed_Holidays_shift_5",
        "Smoothed_Holidays_shift_6",
        "period_1_lower_bound",
        "period_2_lower_bound",
        "period_3_lower_bound",
        "period_4_lower_bound",
        "period_5_lower_bound",
        "period_6_lower_bound",
        "period_1_upper_bound",
        "period_2_upper_bound",
        "period_3_upper_bound",
        "period_4_upper_bound",
        "period_5_upper_bound",
        "period_6_upper_bound",
        "less12",
        "RENTAL_BILLED_QTY",
        "PRODUCT_SALES",
        "RENTAL_SALES",
        "cv2",
        "cv2_sales",
        "RENTAL_SALES_lag_1",
        "RENTAL_SALES_lag_2",
        "RENTAL_SALES_lag_3",
        "RENTAL_SALES_lag_4",
        "RENTAL_SALES_lag_5",
        "RENTAL_SALES_lag_6",
        "SALE_QTY_lag_1",
        "SALE_QTY_lag_2",
        "SALE_QTY_lag_3",
        "SALE_QTY_lag_4",
        "SALE_QTY_lag_5",
        "SALE_QTY_lag_6",
        "DELIVERY",
        "MATERIAL_020112_SALE",
        "MATERIAL_020110_SALE",
        "MATERIAL_020104_SALE",
    ]

    X = df[variables]
    y1 = df["CCH_shift_1"]
    y2 = df["CCH_shift_2"]
    y3 = df["CCH_shift_3"]
    y4 = df["CCH_shift_4"]
    y5 = df["CCH_shift_5"]
    y6 = df["CCH_shift_6"]
    def prediction_lgb(X, y1):
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.25)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # params = {"boosting_type": "dart", "objective": "mse","learning_rate": 0.02,"max_depth": 3,"num_leaves": 12}
        params = {"boosting_type": "dart",
                "objective": "mse",
                'is_unbalance': True,
                'metric':['l1','l2'],
                "tree":"voting",
                "learning_rate": 0.015,
                'num_iterations':10000,
                'early_stopping_round':25,
                'max_bin':600,
                # "max_depth": 3,
                # "bagging_fraction":0.6,
                'feature_fraction':0.8,
                "subsample_freq" : 8,
                "data_sample_strategy":'goss',
                # "num_leaves": 12
                'force_row_wise':True
                }
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=2500,
            valid_sets=lgb_test,
        )
        return gbm
    y1_pred = prediction_lgb(X, y1)
    y2_pred = prediction_lgb(X, y2)
    y3_pred = prediction_lgb(X, y3)
    y4_pred = prediction_lgb(X, y4)
    y5_pred = prediction_lgb(X, y5)
    y6_pred = prediction_lgb(X, y6)

    model = [y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred]

    print(mean_squared_error(y1, model[0].predict(X)))
    print(mean_squared_error(y3, model[2].predict(X)))
    print(mean_squared_error(y6, model[5].predict(X)))

    print(mean_squared_error(y3, model_class[2].predict(X)))


    with open("model_list6_202406.pkl", "wb") as f:
        dill.dump(model, f)
    with open("model_list6_2024.pkl", "rb") as f:
        model_class = dill.load(f)
    # with open("model_list6.pkl", "rb") as f:
    #     model = dill.load(f)
    return model

def calculate_slope(row):
    data = [
        # row["CCH_lag_2"],
        row["CCH_lag_1"],
        row["CCH"],
        row["pred1"],
        row["pred2"],
        row["pred3"],
        # row["pred4"],
        # row["pred5"],
        # row["pred6"],
    ]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(data)), data
    )
    if slope < 0:
        period_to_cross_zero = abs(intercept / slope)
    else:
        period_to_cross_zero = None
    # max_cch = max([row["CCH_lag_2"], row["CCH_lag_1"], row["CCH"]])
    max_cch = max([row["CCH_lag_1"], row["CCH"]])
    if max_cch != 0:
        percent_decline = slope / max_cch
    else:
        percent_decline = 0

    drop = row["CCH_lag_1"] - row["CCH"]

    return slope, percent_decline, max_cch, p_value, period_to_cross_zero, drop


def predict_classifer(df_long, df_wide, model,
                    start_date="2022-06-01"):
    tqdm.pandas()
    df_wide["POSTING_DATE"] = df_wide["PREDICTION_DATE"]
    df_wide["POSTING_DATE"] = pd.to_datetime(df_wide["POSTING_DATE"])
    df_long["POSTING_DATE"] = pd.to_datetime(df_long["POSTING_DATE"])
    df_wide = df_wide[df_wide["POSTING_DATE"] == start_date]
    df_long = df_long[df_long["POSTING_DATE"] == df_long["POSTING_DATE"].max()]
    df_long.drop(['POSTING_DATE'], axis=1, inplace=True)
    df_wide = df_wide.merge(df_long, on=["CUSTOMER_SHIPTO"], how='inner')
    ### Combining the fields()
    for period in tqdm(range(1, 7)):
        column_name = f"period_{period}_uncertainty"
        upper_bound = df_wide[f"period_{period}_upper_bound"]
        lower_bound = df_wide[f"period_{period}_lower_bound"]
        df_wide[column_name] = upper_bound - lower_bound

    X = df_wide[variables]

    df_wide["pred1"] = model[0].predict(X)
    df_wide["pred2"] = model[1].predict(X)
    df_wide["pred3"] = model[2].predict(X)
    df_wide["pred4"] = model[3].predict(X)
    df_wide["pred5"] = model[4].predict(X)
    df_wide["pred6"] = model[5].predict(X)

    df_wide[["slope", "percent_decline", "max_cch", "p_value", "cross_zero", "drop"]] = df_wide.apply(
        calculate_slope, axis=1, result_type="expand"
    )
    # df_out['explainer'] = X.progress_apply(lambda row: explainer_d(gbm_explainer, row), axis=1)

    return df_wide

def explainer(df_wide, gbm_explainer):
    X = df_wide[variables]
    def conditional_apply(row):
        corresponding_row = df_wide.loc[row.name]
        if (corresponding_row['max_cch'] >= 0):
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
    with open("gbm_explainer_202406.pkl", "wb") as f:
        dill.dump(gbm_explainer, f)
    return gbm_explainer

def explainer_d(gbm_explainer, instance):
    # select a single Row
    # instance = X.iloc[0]
    prediction_breakdown = gbm_explainer.predict_parts(instance, B=5, N=10, type='break_down').result
    # print(prediction_breakdown["variable_name"].unique())
    replacements = {
        r"period_\d+_": "",
        r"_lag_\d": "",
        r"_shift_\d": "",
        r"Smoothed_Holidays": "Holidays",
        r"cv2": "Cyclicality",
        r"_\d": "",
        r"sample_mean": "Trend",
        r"ROLL_MEAN": "CCH",
        r"CCH": "CCH",
        r"uncertainty": "Irregularity",
        r"upper_bound": "Irregularity",
        r"lower_bound": "Irregularity",
        r"DISCOUNT_RATIO": "Discount",
        r"CNTD_RENTAL_POSTING_DATE": "Rental_Freq",
        r"CNTD_POSTING_DATE": "Posting_Freq",
        r"CNT_POSTING_DATE": "Posting_Freq",
        r"PROD_LRGLG_SALE": "Product_Type",
        r"PROD_5MEDLE_SALE": "Product_Type",
        r"PROD_1MEDLE_SALE": "Product_Type",
        r"PROD_SMLLD_SALE": "Product_Type",
        r"PROD_4MEDLE2_SALE": "Product_Type",
        r"PROD_8LRGLG_SALE": "Product_Type",
        r"MATERIAL_020112_SALE": "Product_Type",
        r"MATERIAL_050299_SALE": "Product_Type",
        r"MATERIAL_020110_SALE": "Product_Type",
        r"MATERIAL_020104_SALE": "Product_Type",
        r"MATERIAL_111899_SALE": "Product_Type",
        r"MATERIAL_051299_SALE": "Product_Type",
        r"PROD_SMLLD2_SALE": "Product_Type",
        r"PRODLRGLG_SALE": "Product_Type",
        r"PRODMEDLE2_SALE": "Product_Type",
        r"MATERIAL50299_SALE": "Product_Type",
        r"MATERIAL20110_SALE": "Product_Type",
        r"MATERIAL51299_SALE": "Product_Type",
        r"MATERIAL20104_SALE": "Product_Type",
        r"MATERIAL11899_SALE": "Product_Type",
        r"MATERIAL20112_SALE": "Product_Type",
        r"RENTAL_BILLED_QTY": "Rental",
        r"less12": "lesser_data",
        r"PRODMEDLE_SALE": "Product_Type",
        r"KSD": "Missed_Purchases",
        r"KS_filtered": "Missed_Purchases",
        r"Smoothed_Holidays" r"RENTAL_BILLED_QTY": "Rental",
        r"DELIVERY": "Delivery",
        r"RENTAL_SALES": "Rental",
        r"DAILY_RENT": "Rent_Collect_Period",
        r"MONTHLY_RENT": "Rent_Collect_Period",
        r"QUARTERLY_RENT": "Rent_Collect_Period",
        r"ANNUAL_RENT": "Rent_Collection_Freq",
        r"Other_Rent_Period": "Rent_Collect_Period",
        r"REFERENCE_ITEMS": "Interactions",
        r"POSTING_PERIOD": "Interactions",
        r"CNT_POSTING_DATE": "Interactions",
        r"DAY_BETWEEN_POSTING": "Interactions",
        r"AVG_DOCUMENT_ISSUE_DIFF": "Delays",
        r"AVG_POST_ISSUE_DIFF": "Delays",
        r"PRODUCT_SALES": "Sales",
        r"SALE_QTY": "Sales",
    }
    # i want to replace all iteratively
    prediction_breakdown["variable_name"] = prediction_breakdown[
        "variable_name"
    ].replace(replacements, regex=True)
    prediction_breakdown['variable_name'] = prediction_breakdown['variable_name'].replace(replacements, regex=True)
    prediction_breakdown = prediction_breakdown.groupby('variable_name')['contribution'].sum().abs().reset_index()
    prediction_breakdown = prediction_breakdown[
        ~prediction_breakdown["variable_name"].str.contains(
            '^$|intercept|Trend|Uncertainty|Holidays|""'
        )
    ]
    top_contributions = prediction_breakdown.nlargest(3, 'contribution')['variable_name'].tolist()
    string = f"This is due to {top_contributions[0]}, {top_contributions[1]}, and {top_contributions[2]}. "

    return string

# for i in range(108260,108270):
#     print(explainer_d(gbm_explainer,X.iloc[i]))
# df_new = df_wide[["CUSTOMER_SHIPTO","pred1","CCH_shift_1","pred2","CCH_shift_2","pred3","CCH_shift_3","pred4","CCH_shift_4","pred5","CCH_shift_5","pred6","CCH_shift_6"]]
