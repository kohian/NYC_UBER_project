import pandas as pd
from google.cloud import bigquery

from nyc_forecasting.inference.config import PipeConfig, BigQueryConfig, XGBoostInferConfig
from nyc_forecasting.inference.bigquery_io import load_dataframe_to_bigquery, load_latest_timestamp_from_bigquery

from nyc_forecasting.bigquery_sql.run_bigquery_sql import run_prediction_error_merge

from nyc_forecasting.core.data import (
    load_monthly_files,
    select_files,
)

def main() -> None:
    pipe_cfg = PipeConfig()
    bigquery_config = BigQueryConfig()
    model_config = XGBoostInferConfig()

    all_files = select_files(
        pipe_cfg.pipe_src,
        pipe_cfg.single_pipe_start_end,
        pipe_cfg.single_pipe_start_end,
    )

    print("All files:", all_files)

    df = load_monthly_files(all_files)

    client = bigquery.Client(project=bigquery_config.project_id)

    latest_timestamp = load_latest_timestamp_from_bigquery(client = client, table_id = bigquery_config.demand_actuals_table)
    next_timestamp = latest_timestamp  + pd.Timedelta(hours=1)
    if next_timestamp.tzinfo is not None:
        next_timestamp = next_timestamp.tz_localize(None)
    
    # filter for next_timestamp
    df["hour"] = pd.to_datetime(df["hour"])
    df = df[df["hour"] == next_timestamp]
    if df.empty:
        raise ValueError(f"No rows found for next_timestamp={next_timestamp}")

    df["PULocationID"] = df["PULocationID"].astype(str)

    ACTUALS_SCHEMA = [
        bigquery.SchemaField("hour", "TIMESTAMP"),
        bigquery.SchemaField("PULocationID", "STRING"),
        bigquery.SchemaField("demand", "FLOAT"),
    ]

    load_dataframe_to_bigquery(
        client = client,
        df = df,
        table_id = bigquery_config.demand_actuals_table,
        schema = ACTUALS_SCHEMA,
        write_disposition = "WRITE_APPEND", # i want to append! 
    )

    run_prediction_error_merge(
        client = client,
        target_timestamp= next_timestamp,
        model_version= model_config.model_version
    )

if __name__ == "__main__":
    main()

