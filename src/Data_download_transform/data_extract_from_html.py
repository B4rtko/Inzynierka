import os
import json
from bs4 import BeautifulSoup
import pandas as pd

item_container_div_class = "pane-legend-item-value-container"
item_title_span_class = "pane-legend-item-value-title pane-legend-line pane-legend-item-value-title__main"
item_value_span_class = "pane-legend-item-value pane-legend-line pane-legend-item-value__main"
item_value_volume_span_class = "pane-legend-item-value pane-legend-line"


def _load_from_json(instrument_name):
    with open(os.path.join('Data', '1_Raw', instrument_name + '.json'), 'r', encoding='utf-8') as f:
        data_raw_dict = json.load(f)

    return list(data_raw_dict.values())


def _extract_item_title(item_container_soup):
    return [item.text for item in item_container_soup \
        .find_all("div", class_=item_container_div_class)[0] \
        .find_all("span", class_=item_title_span_class)
            ]


def _extract_item_value(item_container_soup):
    return [item.text for item in item_container_soup \
        .find_all("div", class_=item_container_div_class)[0] \
        .find_all("span", class_=item_value_span_class)
            ]


def _extract_item_value_volume(item_container_soup):
    return item_container_soup \
        .find_all("div", class_=item_container_div_class)[1] \
        .find_all("span", class_=item_value_volume_span_class)[0].text


def _dataframe_K_to_1000(
        df: pd.DataFrame,
        column: str,
) -> pd.DataFrame:
    df = df.copy()

    flag_K = ["K" in str(i) for i in df[column]]
    flag_M = ["M" in str(i) for i in df[column]]

    df[column] = df[column].apply(lambda x: float(str(x).replace("K", "").replace("M", "")))

    df[column][flag_K] *= 1_000
    df[column][flag_M] *= 1_000_000

    return df.astype('float64')


def _dataframe_remove_last_duplicates(df):
    first_duplicate = df[
        (
                (df.diff() == [0, 0, 0, 0, 0]) &
                (df.diff(2) == [0, 0, 0, 0, 0]) &
                (df.diff(3) == [0, 0, 0, 0, 0]) &
                (df.diff(4) == [0, 0, 0, 0, 0]) &
                (df.diff(5) == [0, 0, 0, 0, 0]) &
                (df.diff(6) == [0, 0, 0, 0, 0]) &
                (df.diff(7) == [0, 0, 0, 0, 0]) &
                (df.diff(8) == [0, 0, 0, 0, 0]) &
                (df.diff(9) == [0, 0, 0, 0, 0]) &
                (df.diff(10) == [0, 0, 0, 0, 0]) &
                (df.diff(11) == [0, 0, 0, 0, 0]) &
                (df.diff(12) == [0, 0, 0, 0, 0]) &
                (df.diff(13) == [0, 0, 0, 0, 0]) &
                (df.diff(14) == [0, 0, 0, 0, 0]) &
                (df.diff(15) == [0, 0, 0, 0, 0]) &
                (df.diff(16) == [0, 0, 0, 0, 0]) &
                (df.diff(17) == [0, 0, 0, 0, 0]) &
                (df.diff(18) == [0, 0, 0, 0, 0]) &
                (df.diff(19) == [0, 0, 0, 0, 0]) &
                (df.diff(20) == [0, 0, 0, 0, 0])
        ).iloc[:, 0]
    ].index[0] - 20

    return df.loc[:first_duplicate, :].copy()


def extract_data_to_dataframe(
        instrument_name: str,
        save_to_csv: bool = True,
):
    raw_list = _load_from_json(instrument_name)
    raw_soup_list = [BeautifulSoup(i) for i in raw_list]

    columns = _extract_item_title(raw_soup_list[0]) + ["Volume"]
    records = []

    for i in raw_soup_list:
        try:
            records.append(_extract_item_value(i) + [_extract_item_value_volume(i)])
        except:
            records.append([None for _ in range(len(columns))])

    df = pd.DataFrame(records, columns=columns).dropna()
    df = _dataframe_K_to_1000(df, "Volume")
    df = _dataframe_remove_last_duplicates(df)

    if save_to_csv:
        df.to_csv(os.path.join('Data', '2_Extracted', instrument_name + '.csv'))

    return df


if __name__ == "__main__":
    instrument_name_list = [
        # "Crude_Oil_5",
        # "Tesla_5",
        # "DJI_5",
        # "Apple_5",
        "Amazon_5",
    ]
    
    for instrument in instrument_name_list:
        extract_data_to_dataframe(instrument, True)

