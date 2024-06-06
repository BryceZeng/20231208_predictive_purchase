from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sqlalchemy import create_engine, text
import dill
import numpy as np
import pandas as pd
from scipy import stats
import pyodbc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import polars as pl
import re

def get_data(sql):
    conn = pyodbc.connect(
        """DRIVER={ODBC Driver 17 for SQL Server};
        SERVER=mlggae00sql005.linde.lds.grp,1433;
        DATABASE=APAC_DATA_REPO;
        UID=Driver_Safety_PRD;
        PWD=HYKhEIMgj0rDvhrl;
        Encrypt=no;
        TrustServerCertificate=no;
        Connection Timeout=300;
        ColumnEncryption=Enabled;
        DisableCertificateVerification=yes;"""
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


for j in tqdm(range(2, 6)):
    T0_m = datetime.today() - relativedelta(months=j)
    T0_m = T0_m.replace(day=1)
    for i in range(1, 19):
        # create new variable
        globals()['T' + str(i) + '_m'] = T0_m - relativedelta(months=i)
        globals()['T' + str(i) + '_m'] = globals()['T' + str(i) + '_m'].replace(day=1)
    for i in range(0, 19):
        globals()['T' + str(i) + '_m'] = globals()['T' + str(i) + '_m'].strftime("%Y-%m") + "-01"

    fd = open(r'sql_bundle.sql', 'r')
    sqlFile = fd.read()
    fd.close()

    for i in range(0, 19):
        sqlFile = re.sub('TMINUS'+ str(i) + '_TIME_SET', str(globals()['T' + str(i) + '_m']), sqlFile)
    df = get_data(sqlFile)
    df.to_pickle(f'df_{j}.pkl')

"""
## Main Data Exploration
"""
df.describe()
df.info()
"""
creating a dummy value for if a customer will purchase additional different products
"""
df['purchase'] = df['NEW_MATNR'].apply(lambda x: 1 if x != 0 else 0)

plt.hist(target)
plt.title('Target distribution')