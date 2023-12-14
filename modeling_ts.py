from concurrent.futures import ThreadPoolExecutor
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
# from gluonts.mx import DeepAREstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd


def create_gluonts_dataset(df):
    def group_to_dict(x):
        return {
            FieldName.TARGET: x["CCH"].values,
            FieldName.START: x["POSTING_DATE"].min(),
            FieldName.ITEM_ID: str(x.name),
        }

    with tqdm(total=df["CUSTOMER_SHIPTO"].nunique()) as pbar:

        def wrapped_group_to_dict(x):
            pbar.update()
            return group_to_dict(x)

        grouped = df.groupby("CUSTOMER_SHIPTO").apply(wrapped_group_to_dict)
    return grouped.to_list()


# ---------------------------------------#


def train_timeseries(df):
    df["POSTING_DATE"] = pd.to_datetime(df["POSTING_DATE"])
    data = create_gluonts_dataset(df)
    training_data = ListDataset(data, freq="M")
    estimator = DeepAREstimator(
        freq="MS",
        prediction_length=6,
        num_layers=3,
        num_cells=70,
        dropout_rate=0.25,
        trainer=Trainer(epochs=23, num_batches_per_epoch=3000, learning_rate=5e-5),
    )
    model = estimator.train(training_data=training_data)
    return model


def predict_timeseries(df, model, start_date="2022-06-01"):
    df_subset = df[df["POSTING_DATE"] <= start_date]
    df_subset = df_subset[
        df_subset["CUSTOMER_SHIPTO"].isin(
            df_subset[df_subset["CCH"] != 0]
            .groupby("CUSTOMER_SHIPTO")["POSTING_DATE"]
            .count()
            .loc[lambda x: x >= 3]
            .index
        )
    ]
    data_test = create_gluonts_dataset(df_subset)
    test_data = ListDataset(data_test, freq="1M")

    # Generate forecasts
    forecasts = list(model.predict(test_data))

    dfs = [
        pd.DataFrame(
            {
                "CUSTOMER_SHIPTO": forecast.item_id,
                "PERIOD": [f"period_{i+1}" for i in range(forecast.samples.shape[1])],
                "sample_mean": forecast.samples.mean(axis=0),
                "lower_bound": np.percentile(forecast.samples, 25, axis=0),
                "upper_bound": np.percentile(forecast.samples, 75, axis=0),
            }
        )
        for forecast in tqdm(forecasts)
    ]
    forecast_df = pd.concat(dfs, ignore_index=True)
    df_wide = forecast_df.pivot_table(
        index="CUSTOMER_SHIPTO",
        columns="PERIOD",
        values=["sample_mean", "lower_bound", "upper_bound"],
    )
    df_wide.columns = df_wide.columns.map("{0[1]}_{0[0]}".format)
    df_wide["PREDICTION_DATE"] = start_date
    return df_wide
