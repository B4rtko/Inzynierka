import os
import json
from bs4 import BeautifulSoup
import pandas as pd

with open('../../Data/Raw/Crude_Oil_5.json', 'r', encoding='utf-8') as f:
    data_raw_dict = json.load(f)
    data_raw_list = list(data_raw_dict.values())

item_container_div_class = "pane-legend-item-value-container"
item_title_span_class = "pane-legend-item-value-title pane-legend-line pane-legend-item-value-title__main"
item_value_span_class = "pane-legend-item-value pane-legend-line pane-legend-item-value__main"
item_value_volume_span_class = "pane-legend-item-value pane-legend-line"


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


def extract_data_to_dataframe(raw_list):
    raw_soup_list = [BeautifulSoup(i) for i in raw_list]

    columns = _extract_item_title(raw_soup_list[0]) + ["Volume"]
    records = []

    for i in raw_soup_list:
        try:
            records.append(_extract_item_value(i) + [_extract_item_value_volume(i)])
        except:
            records.append([None for _ in range(len(columns))])

    return pd.DataFrame(records, columns=columns)


df = extract_data_to_dataframe(data_raw_list)