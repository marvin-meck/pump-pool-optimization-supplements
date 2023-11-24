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
import pyomo.environ as pyo

from model_set_cover import pyomo_create_model as set_cover
from model_maximum_coverage import pyomo_create_model as max_coverage

HOME = Path(__file__).resolve().parent

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="INPUT", type=Path, required=True)
    parser.add_argument("-o", "--output", dest="OUTPUT", type=Path, required=True)
    
    parser.add_argument("--solver", dest="SOLVER", type=str, required=True)

    args = parser.parse_args()

    coverage = []

    sc = set_cover().create_instance(filename=args.INPUT.as_posix())
    mc = max_coverage()

    dp = pyo.DataPortal(model=mc)
    dp.load(filename=args.INPUT.as_posix())
    dp.__setitem__("max_num_sets",{None:1})
    # dp.store()
    instance = mc.create_instance(dp)


    with pyo.SolverFactory(args.SOLVER) as opt:
        opt.solver_io="python"
        results = opt.solve(sc, tee=False)
    
        if (results.solver.termination_condition == pyo.TerminationCondition.optimal):
            upper = int(pyo.value(sc.cardinality))
        else:
            raise Warning("solver terminated with termination condition {}... returning".format(results.solver.termination_condition))

        for max_num_sets in range(1,upper):
            dp.__setitem__("max_num_sets",{None:max_num_sets})
            instance.max_num_sets = max_num_sets
            
            opt.solve(instance, tee=False, warmstart=True)
            if (results.solver.termination_condition == pyo.TerminationCondition.optimal):
                coverage.append( pyo.value(instance.coverage) / len(instance.Elements))
            else:
                raise Warning("solver terminated with termination condition {}... returning".format(results.solver.termination_condition))


    df = pd.DataFrame({"max_num_sets":range(1,len(coverage)+1),"rel_coverage":coverage})
    df.to_csv(args.OUTPUT, index=False)