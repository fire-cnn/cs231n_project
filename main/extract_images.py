""" 
Extract images from the NAIP collection in the Planetary Computer

This extracts the NAIP data from the PC for a pre-defined shapefile. This relies
on the NAIP class and needs authentication in the PC SDK API. More info can be
found in the API docs: https://pypi.org/project/planetary-computer/
"""

import argparse

from src.naip import NAIP


def main(aoi_path, save_path, date_window_up, date_window_down):
    date_window = (date_window_down, date_window_up)
    naip = NAIP(aoi_path=aoi_path, save_path=save_path, date_window=date_window)
    naip.execute_search_aoi()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date_window_down", type=int)
    parser.add_argument("--date_window_up", type=int)
    parser.add_argument("--path_to_shape")
    parser.add_argument("--save_file")
    args = parser.parse_args()

    main(args.path_to_shape, args.save_file, args.date_window_up, args.date_window_down)
