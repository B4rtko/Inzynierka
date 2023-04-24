import json
import time
import pyautogui as pag
import pyperclip

from pynput import keyboard
from typing import List


def _scrap_investing_com_step():
    """
    function changes focus to window below (html inspector), copies selected part of html,
        goes back to window above (browser window) and returns element from clipboard.
    :return: Copied html part
    """
    pag.hotkey("win", "down")
    time.sleep(.001)
    pag.hotkey("ctrl", "c")
    time.sleep(.01)
    pag.hotkey("win", "up")
    time.sleep(.001)
    return pyperclip.paste()


def _scrap_investing_com(
        steps: int,
        _data_list: list,
) -> List:
    """
    function walks through 'steps' number of steps of copying data from html inspector and returns list with results.
        function also populates given '_data list' - it is useful when running from jupyter notebook instead of script.
    :param steps: number of iterations of data scraping
    :param _data_list: container to fill
    :return: list populated with scrapped html data
    """
    for i in range(steps):
        if i%100 == 0:
            print(i)
        _data_list.append(_scrap_investing_com_step())
        pag.press("right")
    return _data_list


def scrap_main(
        filename: str = 'data.json'
):
    """
    main function for scraping data from investing.com.
    Prerequisites:
        - works on Linux OS with i3 as window manager, but adjusting function '_scrap_investing_com_step' will allow
            to run it on any other OS
        - open browser on desired stock shares plot, close any adds or pop-up windows
        - add volume indicator to plot; scraping will work with any other indicator or even none of them,
            but data extraction later on is set to work with prices and volume
        - open full-screen plot mode
        - open html inspector in new window below the browser
        - set desired interval and move plot pointer to desired starting date; with 5 minutes interval max data range
            was about 1 year
        - inside inspector you should select 'pane-legend' div; rectangle with prices, volumes and other indicators
            should be highlighted (sign that selected part in inspector leads to those values)
        - click on the plot to set focus (you should be able to move plot with right and left arrow)
        - press right shift to start scrapping process; it cannot be stopped - it must finish the iteration first
            in order to save the result to file
    Scrapped results are saved to json file with convention of dict with keys as element enumeration.
    Can be run from jupyter notebook for using early stopping.

    :param filename: json path to save scrapped results
    """
    data_container = []
    flag = 0
    with keyboard.Events() as events:
        while True:
            for event in events:
                if event.key == keyboard.Key.shift_r:
                    flag = 1
                    time.sleep(1)
                    # add try except on KeyboardInterrupt
                    try:
                        _scrap_investing_com(7_000, data_container)
                    except KeyboardInterrupt:
                        break

            if flag == 1:
                break
    data_dict = {k: v for (k, v) in enumerate(data_container)}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    scrap_main("DJI_5.json")


