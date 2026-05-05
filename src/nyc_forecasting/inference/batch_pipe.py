import pandas as pd
from google.cloud import bigquery

from nyc_forecasting.inference.config import PipeConfig
from nyc_forecasting.inference.bigquery_io import load_dataframe_to_bigquery

from nyc_forecasting.core.data import (
    load_monthly_files,
    select_files,
)

# def upload_to_bigquery(df: pd.DataFrame, table_id: str) -> None:
#     client = bigquery.Client()

#     job_config = bigquery.LoadJobConfig(
#         write_disposition="WRITE_APPEND",  # append mode
#         schema=[
#             bigquery.SchemaField("hour", "TIMESTAMP"),
#             bigquery.SchemaField("PULocationID", "STRING"),
#             bigquery.SchemaField("demand", "FLOAT"),
#         ],
#     )

#     job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
#     job.result()  # wait for completion

#     print(f"Loaded {len(df)} rows into {table_id}")


def main() -> None:
    pipe_cfg = PipeConfig()

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

    client = bigquery.Client()

    load_dataframe_to_bigquery(
        client = client,
        df = df,
        table_id = pipe_cfg.bq_table_id,
        schema = ACTUALS_SCHEMA,
        write_disposition = "WRITE_TRUNCATE", # i want to override! 
    )

    # upload_to_bigquery(
    #     df,
    #     pipe_cfg.bq_table_id,  # e.g. "project.dataset.hourly_demand_actuals"
    # )


if __name__ == "__main__":
    main()

