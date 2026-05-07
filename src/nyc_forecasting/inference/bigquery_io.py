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


def load_all_actuals_from_bigquery(
    client: bigquery.Client,
    actuals_table: str,
) -> pd.DataFrame:
    query = f"""
    SELECT
        hour,
        PULocationID,
        demand
    FROM `{actuals_table}`
    ORDER BY hour, PULocationID
    """

    return client.query(query).to_dataframe()


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


def load_latest_timestamp_from_bigquery(
    client: bigquery.Client,
    table_id: str,
) -> pd.Timestamp:
    query = f"""
    SELECT
        MAX(hour) AS latest_hour
    FROM `{actuals_table}`
    """

    result = client.query(query).to_dataframe()

    if result.empty or pd.isna(result.loc[0, "latest_hour"]):
        raise ValueError(f"No timestamps found in {actuals_table}")

    return pd.Timestamp(result.loc[0, "latest_hour"])


# MAY NOT WANT THIS OR NEED A DIFFERENT VERSION OF THIS

def prediction_already_exists(
    client: bigquery.Client,
    predictions_table: str,
    target_timestamp: pd.Timestamp,
    model_version: str,
) -> bool:
    query = f"""
    SELECT COUNT(*) AS cnt
    FROM `{predictions_table}`
    WHERE target_timestamp = @target_timestamp
      AND model_version = @model_version
    """

    # This part handles the @ in the query which prevents SQL injection
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "target_timestamp",
                "TIMESTAMP",
                target_timestamp.to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    )

    result = client.query(query, job_config=job_config).to_dataframe()
    return int(result.loc[0, "cnt"]) > 0
