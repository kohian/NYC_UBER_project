from pathlib import Path
import pandas as pd
from google.cloud import bigquery

PREDICTION_ERROR_MERGE_SQL_PATH =  Path(__file__).resolve().parent / "prediction_error_merge.sql"

def run_prediction_error_merge(
    client: bigquery.Client,
    target_timestamp: pd.Timestamp,
    model_version: str,
) -> None:
    query = PREDICTION_ERROR_MERGE_SQL_PATH.read_text()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "target_timestamp",
                "TIMESTAMP",
                pd.Timestamp(target_timestamp).to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter(
                "model_version",
                "STRING",
                model_version,
            ),
        ]
    )

    job = client.query(query, job_config=job_config)
    job.result()

    print(
        f"Updated prediction error table for "
        f"target_timestamp={target_timestamp}, model_version={model_version}"
    )