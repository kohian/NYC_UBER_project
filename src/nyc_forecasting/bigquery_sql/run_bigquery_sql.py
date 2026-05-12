from pathlib import Path
import pandas as pd
from google.cloud import bigquery

SQL_DIR = Path(__file__).resolve().parent


def run_parameterized_merge_sql(
    client: bigquery.Client,
    sql_filename: str,
    target_timestamp: pd.Timestamp,
    model_version: str,
) -> None:
    query_path = SQL_DIR / sql_filename
    query = query_path.read_text()

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
        f"Ran {sql_filename} for "
        f"target_timestamp={target_timestamp}, model_version={model_version}"
    )




# MERGE_PREDICTION_ERROR_SQL_PATH =  Path(__file__).resolve().parent / "merge_prediction_error.sql"

# def run_merge_prediction_error(
#     client: bigquery.Client,
#     target_timestamp: pd.Timestamp,
#     model_version: str,
# ) -> None:
#     query = MERGE_PREDICTION_ERROR_SQL_PATH.read_text()

#     job_config = bigquery.QueryJobConfig(
#         query_parameters=[
#             bigquery.ScalarQueryParameter(
#                 "target_timestamp",
#                 "TIMESTAMP",
#                 pd.Timestamp(target_timestamp).to_pydatetime(),
#             ),
#             bigquery.ScalarQueryParameter(
#                 "model_version",
#                 "STRING",
#                 model_version,
#             ),
#         ]
#     )

#     job = client.query(query, job_config=job_config)
#     job.result()

#     print(
#         f"Updated prediction error table for "
#         f"target_timestamp={target_timestamp}, model_version={model_version}"
#     )