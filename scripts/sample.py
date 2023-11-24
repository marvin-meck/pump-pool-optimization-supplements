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
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import pyodbc
from sklearn.neighbors import KernelDensity


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""This is a preprocessing module. It\n
            i. takes a data source (database or file) (expects to find `flow_rate`, `head` and `kin_viscosity`),\n
            ii. creates a KDE in log-scale in three dimensions (`flow_rate`, `head` and `kin_viscosity`; given in m3/h, m, and cSt respectively),\n
            iii. samples from the KDE and writes the sample to disk\n
        """
    )
    
    parser.add_argument("-c", "--config", dest='CONFIG', required=False, type=Path, help='config file')
    parser.add_argument("-f", "--file", dest="IN_FNAME", required=False, type=Path, help='input file')
    parser.add_argument("-m", "--mode", dest='MODE', required=True, type=str, help='input format')
    
    parser.add_argument("-q", "--query", dest='QUERY', required=False, default=None, type=Path, help='query to send to db')
    parser.add_argument("-o", "--output", dest="OUT_FNAME", required=False, type=Path, help="name of the output file")
    parser.add_argument("-n", "--num-sample", dest="NSAMPLE", required=False, default=1000, type=int, help="size of the sample")

    if __debug__:
        parser.set_defaults(
            QUERY=Path(r"./.config/query.sql"),
            CONFIG=Path(r"./.config/pyodbc.yaml"),
            OUT_FNAME= Path(r"./out") / "data_{}_sample.csv".format(datetime.today().strftime('%y%m%d'))
        )

    args = parser.parse_args()

    if args.MODE == "odbc":

        if not args.CONFIG is None:
            if not args.CONFIG.exists():
                raise IOError("file {} not found or does not exist".format(args.CONFIG))
    
            with open(args.CONFIG, "r") as f:
                tmp = yaml.safe_load(f)
                try:
                    DRIVER = tmp["driver"]
                    DBFILE = tmp["dbfile"]
                except KeyError:
                    raise IOError("invalid config file {} for mode ".format(args.CONFIG, args.MODE))
        else: 
            raise IOError("missing config file")
        
        cnxn = pyodbc.connect("Driver={{{0}}};DBQ={1};".format(DRIVER,DBFILE))

        if not args.QUERY is None:
            with open(args.QUERY,'r') as istream:
                query = istream.read()
                data = pd.read_sql(query,cnxn).dropna()
        else: 
            raise IOError("no query provided!")
    elif args.MODE == "csv":
        if not args.IN_FNAME is None:
            if not args.CONFIG.exists():
                raise IOError("file {} not found or does not exist".format(args.IN_FNAME))
        else: 
            raise IOError("missing input file")

        data = pd.read_csv(args.IN_FNAME.as_posix())
    else: 
        raise NotImplementedError("input mode {} not implemented".format(args.MODE))
    
    if not args.OUT_FNAME.exists():
        print("creating a new sample")

        X = np.vstack([\
            np.log(data["flow_rate"]),\
            np.log(data["head"]),\
            np.log(data["kin_viscosity"]),\
        ])
    
        d = X.shape[0]
        n = X.shape[1]
        # bw = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
        bw = n**(-1./(d+4)) # scott
        
        kde = KernelDensity(bandwidth=bw, metric='euclidean',
                            kernel='gaussian', algorithm='ball_tree')
        
        kde.fit(X.T)
        log_sample = kde.sample(args.NSAMPLE)
        
        sample = pd.DataFrame(
            data=np.exp(log_sample),
            columns=["flow_rate","head","kin_viscosity"]
        )
        sample.to_csv(args.OUT_FNAME, index_label="element_id")