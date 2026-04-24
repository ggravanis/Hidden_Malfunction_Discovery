from pathlib import Path

import pandas as pd


RAW_FULL_COLUMN_COUNT = 47
SELECTED_RAW_POSITIONS = list(range(2, 16)) + list(range(17, 47))
WORKING_COLUMN_COUNT = len(SELECTED_RAW_POSITIONS)

LABEL_COL = "col_label"
YEAR_COL = "col_year"
DATETIME_COL = "col_datetime"


def _dummy_names(count):
    return [f"col_{idx:03d}" for idx in range(count)]


FULL_DUMMY_COLUMNS = _dummy_names(RAW_FULL_COLUMN_COUNT)
WORKING_DUMMY_COLUMNS = [FULL_DUMMY_COLUMNS[idx] for idx in SELECTED_RAW_POSITIONS]

DAY_COL = WORKING_DUMMY_COLUMNS[0]
HOUR_COL = WORKING_DUMMY_COLUMNS[1]
MINUTE_COL = WORKING_DUMMY_COLUMNS[2]
MONTH_COL = WORKING_DUMMY_COLUMNS[3]
SECOND_COL = WORKING_DUMMY_COLUMNS[4]
GCUP_COL = WORKING_DUMMY_COLUMNS[16]
POSITION_COL = WORKING_DUMMY_COLUMNS[20]
EXTRA_COUNTER_COLUMNS = WORKING_DUMMY_COLUMNS[6:14]


def _persist_with_same_format(df_raw, file_path):
    if file_path.suffix.lower() == ".zip":
        archive_name = f"{file_path.stem}.csv"
        df_raw.to_csv(
            file_path,
            index=False,
            compression={"method": "zip", "archive_name": archive_name},
        )
    else:
        df_raw.to_csv(file_path, index=False)


def _load_raw_file(file_path):
    if file_path.suffix.lower() == ".zip":
        return pd.read_csv(file_path, compression="zip", low_memory=False)
    return pd.read_csv(file_path, low_memory=False)


def _load_group_dataframe(group, raw_dir):
    zip_path = raw_dir / f"group_{group}.zip"
    csv_path = raw_dir / f"group_{group}.csv"

    if zip_path.exists():
        source_path = zip_path
    elif csv_path.exists():
        source_path = csv_path
    else:
        raise FileNotFoundError(
            f"Could not find raw file for group {group}. Expected {zip_path} or {csv_path}."
        )

    raw_df = _load_raw_file(source_path)

    expected_dummy_headers = _dummy_names(raw_df.shape[1])
    if list(raw_df.columns) != expected_dummy_headers:
        raw_df.columns = expected_dummy_headers
        _persist_with_same_format(raw_df, source_path)

    if raw_df.shape[1] == RAW_FULL_COLUMN_COUNT:
        return raw_df[WORKING_DUMMY_COLUMNS].copy()
    if raw_df.shape[1] == WORKING_COLUMN_COUNT:
        working_df = raw_df.copy()
        working_df.columns = WORKING_DUMMY_COLUMNS
        return working_df

    raise ValueError(
        f"Unexpected raw column count: {raw_df.shape[1]}. Expected {RAW_FULL_COLUMN_COUNT} or {WORKING_COLUMN_COUNT}."
    )


def get_previous_index(_df, negative_indexes):
    previous = None
    target_index = negative_indexes[0]
    for item in _df.index.values:
        if item == target_index:
            break
        previous = item
    return previous if previous is not None else target_index


def remove_previous_rows(_df):
    _df = _df.reset_index()
    reset_points = _df[_df[GCUP_COL].diff() < 0].index
    _df[LABEL_COL] = 0

    for reset_point in reset_points:
        if reset_point > 0:
            _df.at[reset_point, POSITION_COL] = _df.at[reset_point - 1, POSITION_COL]
        _df.at[reset_point, LABEL_COL] = 1

    rows_to_drop = reset_points - 1
    rows_to_drop = rows_to_drop[rows_to_drop >= 0]
    _df.drop(index=rows_to_drop, inplace=True)
    return _df


def add_anomaly_label(_df):
    _df = _df.reset_index()
    reset_points = _df[_df[GCUP_COL].diff() < 0].index
    _df[LABEL_COL] = 0
    for reset_point in reset_points:
        _df.at[reset_point, LABEL_COL] = 1
    return _df


def find_negative_values(_df):
    initial_values = [_df[column].iloc[0] for column in EXTRA_COUNTER_COLUMNS]

    for column, init_value in zip(EXTRA_COUNTER_COLUMNS, initial_values):
        _df[column] = _df[column] - init_value

    negative_value_indices = [
        _df[_df[column] < 0].index.values for column in EXTRA_COUNTER_COLUMNS
    ]

    return negative_value_indices, initial_values


def get_batches(_df):
    df_masked = _df.loc[(_df[GCUP_COL] != _df[GCUP_COL].shift()).values]
    masking_limits = df_masked[df_masked[GCUP_COL] == 0].index

    batched_dataframes = []
    small_batch_counter = 0
    normal_batch_counter = 0

    for i, limit in enumerate(masking_limits[:-1]):
        start_date = limit
        try:
            end_date = masking_limits[i + 1]
            df_temp = df_masked.query("index >= @start_date and index < @end_date")
            if df_temp[GCUP_COL].iloc[-1] < 1000:
                small_batch_counter += 1
                continue

            negative_flag = True
            while negative_flag:
                negative_values_index, initial_values = find_negative_values(df_temp)

                if any(indexes.size > 0 for indexes in negative_values_index):
                    candidates = [
                        (indexes[0], idx)
                        for idx, indexes in enumerate(negative_values_index)
                        if indexes.size > 0
                    ]
                    first_negative_index, source_series_idx = min(candidates, key=lambda x: x[0])
                    previous_index = get_previous_index(
                        df_temp, negative_values_index[source_series_idx]
                    )

                    for column, init_value in zip(EXTRA_COUNTER_COLUMNS, initial_values):
                        df_temp.loc[first_negative_index:, column] += (
                            init_value + df_temp.loc[previous_index, column]
                        )
                else:
                    negative_flag = False

            batched_dataframes.append(df_temp)
            normal_batch_counter += 1
        except Exception as exc:
            print(exc)

    print(f"small batch #{small_batch_counter}")
    print(f"normal batch #{normal_batch_counter}")
    return batched_dataframes


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "batch_step_index"
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in [13]:
        df = _load_group_dataframe(group=group, raw_dir=raw_dir)

        df[YEAR_COL] = 2022
        df[DATETIME_COL] = pd.to_datetime(
            {
                "year": df[YEAR_COL],
                "month": df[MONTH_COL],
                "day": df[DAY_COL],
                "hour": df[HOUR_COL],
                "minute": df[MINUTE_COL],
                "second": df[SECOND_COL],
            },
            errors="coerce",
        )

        df = df[df[DATETIME_COL].notna()].copy()
        df.set_index([DATETIME_COL], inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        list_with_batched_dfs = get_batches(_df=df)
        for i, df_data in enumerate(list_with_batched_dfs, start=1):
            df_data = add_anomaly_label(df_data)
            df_data = remove_previous_rows(df_data)
            if DATETIME_COL in df_data.columns:
                df_data.set_index([DATETIME_COL], inplace=True)
            df_data.to_csv(output_dir / f"batch_{i}.csv")
