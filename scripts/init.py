'''
Copyright 2023 Technical University Darmstadt

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Extracts the operating limits defined by PiC-LF03 Appendix A into a .csv file."
    )

    parser.add_argument("-i", "--input", dest='XLS_FILE', required=True, type=Path, help='VCI Excel Tool')
    parser.add_argument("-o", "--output", dest='CSV_FILE', required=False, default="odbc", type=str, help='input format')

    parser.set_defaults(
        XLS_FILE=Path("./data/2020-08-05-lf03-anhang-a-berechnungsblatt.xls"),
        CSV_FILE=Path("./data/2020-08-05-lf03-anhang-a-berechnungsblatt.csv")
    )

    args = parser.parse_args()

    # TODO Checksum test
    xls = pd.ExcelFile(args.XLS_FILE)

    cols = ['inlet diameter [mm]', 'outlet diameter [mm]', 'impeller diameter [mm]',\
           'speed [min-1]', 'nominal flow rate [m3/h]', 'nominal head [mFs]', 'F1',\
           'F2', 'F3', 'F4', 'left x1 [m3/h]', 'left x2 [m3/h]', 'left y1 [mFs]',\
           'left y2 [mFs]', 'right x1 [m3/h]', 'right x2 [m3/h]', 'right y1 [mFs]',\
           'right y2 [mFs]', 'upper x1 [m3/h]', 'upper x2 [m3/h]',\
           'upper y1 [mFs]', 'upper y2 [mFs]', 'lower x1 [m3/h]',\
           'lower x2 [m3/h]', 'lower y1 [mFs]', 'lower y2 [mFs]']


    df = pd.read_excel(xls, sheet_name="Grenzwerte", usecols="B,D,F,I:AF", skiprows=[0,1,2,3,4,5], index_col=4)
    df.columns = cols
    df.index.name = "register"

    df.to_csv(args.CSV_FILE, index=True)