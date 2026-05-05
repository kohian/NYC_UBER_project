import pandas as pd
from google.cloud import bigquery

def load_dataframe_to_bigquery(
    client: bigquery.Client,
    df: pd.DataFrame,
    table_id: str,
    schema: list[bigquery.SchemaField],
    write_disposition: str = "WRITE_APPEND",
) -> None:
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema=schema,
    )

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=job_config,
    )
    job.result()

    print(f"Loaded {len(df)} rows into {table_id}")


def load_latest_actuals_from_bigquery(
    client: bigquery.Client,
    actuals_table: str,
    lookback_hours: int,
) -> pd.DataFrame:
    query = f"""
    SELECT
        hour,
        PULocationID,
        demand
    FROM `{actuals_table}`
    WHERE hour >= TIMESTAMP_SUB(
        (SELECT MAX(hour) FROM `{actuals_table}`),
        INTERVAL @lookback_hours HOUR
    )
    ORDER BY hour, PULocationID
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("lookback_hours", "INT64", lookback_hours)
        ]
    )

    return client.query(query, job_config=job_config).to_dataframe()

