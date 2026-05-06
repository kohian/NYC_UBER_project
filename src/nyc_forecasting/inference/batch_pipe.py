import pandas as pd
from google.cloud import bigquery

from nyc_forecasting.inference.config import PipeConfig, BigQueryConfig
from nyc_forecasting.inference.bigquery_io import load_dataframe_to_bigquery

from nyc_forecasting.core.data import (
    load_monthly_files,
    select_files,
)

def main() -> None:
    pipe_cfg = PipeConfig()
    bigquery_config = BigQueryConfig()

    all_files = select_files(
        pipe_cfg.pipe_src,
        pipe_cfg.batch_pipe_start,
        pipe_cfg.batch_pipe_end,
    )

    print("All files:", all_files)

    df = load_monthly_files(all_files)
    df["PULocationID"] = df["PULocationID"].astype(str)

    ACTUALS_SCHEMA = [
        bigquery.SchemaField("hour", "TIMESTAMP"),
        bigquery.SchemaField("PULocationID", "STRING"),
        bigquery.SchemaField("demand", "FLOAT"),
    ]

    client = bigquery.Client(project=bigquery_config.project_id)

    load_dataframe_to_bigquery(
        client = client,
        df = df,
        table_id = bigquery_config.demand_actuals_table,
        schema = ACTUALS_SCHEMA,
        write_disposition = "WRITE_TRUNCATE", # i want to override! 
    )



if __name__ == "__main__":
    main()

