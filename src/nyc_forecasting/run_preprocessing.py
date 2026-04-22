from nyc_forecasting.core.config import DataConfig
from nyc_forecasting.core.data import read_parquet, read_csv, write_parquet, get_borough_zone_ids, process_monthly_hourly_demand, generate_year_month_list, build_monthly_raw_path, build_processed_path

def main() -> None:
    data_cfg = DataConfig()


    year_month_list = generate_year_month_list(
        data_cfg.process_start,
        data_cfg.process_end,
    )

    zone_lookup = read_csv(data_cfg.zone_lookup_path)

    manhattan_zones = get_borough_zone_ids(zone_lookup, data_cfg.borough)
    print(f"Number of Manhattan zones: {len(manhattan_zones)}")

    for year_month in year_month_list:
        source_path = build_monthly_raw_path(data_cfg.raw_source, year_month)
        print(f"\nProcessing {source_path} ...")

        df = read_parquet(source_path, columns=data_cfg.columns)

        hourly_df = process_monthly_hourly_demand(
            df,
            manhattan_zones,
            year_month,
            data_cfg.keep_license,
        )

        dest_path = build_processed_path(data_cfg.processed_dest, year_month)
        write_parquet(hourly_df, dest_path)

        print(f"Saved: {dest_path}")
        print(f"Rows: {len(hourly_df):,}")
        print(hourly_df.head())


if __name__ == "__main__":
    main()
