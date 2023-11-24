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

from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"./data")

def _covers(pump: pd.Series, flow_rate, head) -> bool:
    
    if (pump["left x1 [m3/h]"]-pump["left x2 [m3/h]"]) > 0:
        left = (pump["left y1 [mFs]"]-pump["left y2 [mFs]"])/(pump["left x1 [m3/h]"]-pump["left x2 [m3/h]"]) \
            * (flow_rate - pump["left x1 [m3/h]"]) + pump["left y1 [mFs]"] >= head
    else: 
        left = pump["left x1 [m3/h]"] <= flow_rate
    
    if (pump["right x1 [m3/h]"]-pump["right x2 [m3/h]"]) > 0:
        right = (pump["right y1 [mFs]"]-pump["right y2 [mFs]"])/(pump["right x1 [m3/h]"]-pump["right x2 [m3/h]"]) \
            * (flow_rate - pump["right x1 [m3/h]"]) + pump["right y1 [mFs]"] <= head
    else:
        right = pump["right x1 [m3/h]"] >= flow_rate
    
    upper = (pump["upper y1 [mFs]"]-pump["upper y2 [mFs]"])/(pump["upper x1 [m3/h]"]-pump["upper x2 [m3/h]"]) \
            * (flow_rate - pump["upper x1 [m3/h]"]) + pump["upper y1 [mFs]"] >= head
    
    lower = (pump["lower y1 [mFs]"]-pump["lower y2 [mFs]"])/(pump["lower x1 [m3/h]"]-pump["lower x2 [m3/h]"]) \
            * (flow_rate - pump["lower x1 [m3/h]"]) + pump["lower y1 [mFs]"] <= head
    
    return left & right & upper & lower


def create_data_dict(elements_frame: pd.DataFrame, sets_frame: pd.DataFrame):

    data_dict = {
        None:{
            "Elements":{None:elements_frame.index.to_list()},
            "Sets":{None:sets_frame.index.to_list()},
            "FeasibleSets":{ the_elem:[ \
                the_set for the_set in sets_frame.index if _covers(
                    sets_frame.loc[the_set],
                    flow_rate=elements_frame.loc[the_elem,"flow_rate"],
                    head=elements_frame.loc[the_elem,"head"],
                )  ] for the_elem in elements_frame.index
            },
        },
    }

    return data_dict


def _write_set(ostream, data, name):
    for key,vals in data[name].items():
        if key is None:
            # regular set
            ostream.write("set {} := ".format(name))
        else:
            # indexed set
            ostream.write("set {}[{}] := ".format(name,key))

        for val in vals:
            ostream.write(f"{val} \n")
        else:
            ostream.write(";\n\n")


if __name__ == "__main__":

    parser = ArgumentParser(description="preprocessing: generates Pyomo Datacommand/ Yaml file for a given set of flow rate,head pairs")

    parser.add_argument("-i", "--input", dest="INPUT", type=Path, required=True)
    parser.add_argument("-o", "--output", dest="OUTPUT", type=Path, required=True)
    parser.add_argument("-f", "--format", dest="FORMAT", type=str, required=False)
    parser.add_argument("-n", "--new", dest="NEW", action=BooleanOptionalAction, required=False, default=False)
    
    args = parser.parse_args()
    
    OUT_DIR = args.OUTPUT.parents[0]
    
    iso = pd.read_csv(
        DATA_DIR / "2020-08-05-lf03-anhang-a-berechnungsblatt.csv",
        sep=",",
        index_col="register"
    )

    sample = pd.read_csv(args.INPUT, index_col="element_id")

    fname = OUT_DIR / "data_{}_elements.csv".format(
        datetime.today().strftime('%y%m%d')
    )


    if (not fname.exists()) or args.NEW:

        # Gülich, Johann Friedrich (2010): Centrifugal Pumps. Berlin, Heidelberg: Springer Berlin Heidelberg. p. 741
        mask_viscosity = sample["kin_viscosity"] < 10
        
        # reject elements that cannot be covered by any set
        mask_coverage = np.array([ any(_covers(iso.iloc[key],flow_rate=sample.loc[idx,"flow_rate"], head=sample.loc[idx,"head"])\
            for key in range(iso.shape[0])) for idx in sample.index])

        mask_total = mask_viscosity & mask_coverage

        sample["mask_viscosity"] = mask_viscosity
        sample["mask_coverage"] = mask_coverage
        sample["mask_total"] = mask_total
        
        print(f"writing new file {fname}")
        sample.to_csv(fname, index_label="element_id")
        
        elements = sample[mask_total]
    else:
        print(f"reading file {fname}")
        sample = pd.read_csv(fname, index_col="element_id")
        elements = sample[sample["mask_total"]]


    data_dict = create_data_dict(elements,iso)

    if args.FORMAT is None:
        format = args.OUTPUT.as_posix().split(".")[-1]
    else:
        format = args.FORMAT

    if format == "dat":
        print(f"writing new file {args.OUTPUT}")
        
        with open(args.OUTPUT, "w+") as f:
            _write_set(f,data_dict[None],"Elements")
            _write_set(f,data_dict[None],"Sets")
            _write_set(f,data_dict[None],"FeasibleSets")
    elif args.FORMAT in ("yml","yaml"):
        raise NotImplementedError
    else:
        raise IOError(f"Unknown output format {format}")

