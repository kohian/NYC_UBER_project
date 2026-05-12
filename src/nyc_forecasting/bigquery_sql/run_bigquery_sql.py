from pathlib import Path
import pandas as pd
from google.cloud import bigquery

MERGE_PREDICTION_ERROR_SQL_PATH =  Path(__file__).resolve().parent / "merge_prediction_error.sql"

def run_merge_prediction_error(
    client: bigquery.Client,
    target_timestamp: pd.Timestamp,
    model_version: str,
) -> None:
    query = MERGE_PREDICTION_ERROR_SQL_PATH.read_text()

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