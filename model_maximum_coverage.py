"""Implementation of the Maximum Coverage Problem [1] 
    using the Pyomo optimization modelling language [2]

Author: Marvin Meck
E-Mail: marvin.meck@fst.tu-darmstadt.de

Corresponding: Prof. Dr.-Ing. Peter Pelz
E-Mail: peter.pelz@fst.tu-darmstadt.de

Funding:
-------
We thank the German Federal Ministry for Economic Affairs and Climate Action (BMWK)
for financially supporting the work within the scope of the HECTOR research project 
as part of the ENPRO 2.0 initiative (funding code: 03EN2006A).

References:
-----------

[1] Hochbaum, Dorit S. (1997): Approximation algorithms for NP-hard problems. 
    Boston, London: PWS Pub. Co.

[2] Hart, William E., William E. Hart; Watson, Jean-Paul; Laird, Carl D.;
    Nicholson, Bethany L.; Siirola, John D. (2017): Pyomo. Optimization
    Modeling in Python. Second edition /  William E Hart [and six others].
    Cham: Springer (Springer Optimization and Its Applications, 67).

Legal Notice
------------
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
"""

import pyomo.environ as pyo


def pyomo_create_model(options=None, model_options=None) -> pyo.AbstractModel:
    """Callback function to create the AbstractModel, see Hart et al. (2017), p.90
    for documentation.

    Hart, William E., William E. Hart; Watson, Jean-Paul; Laird, Carl D.;
    Nicholson, Bethany L.; Siirola, John D. (2017): Pyomo. Optimization
    Modeling in Python. Second edition /  William E Hart [and six others].
    Cham: Springer (Springer Optimization and Its Applications, 67).
    """

    model = pyo.AbstractModel("Maxmium Coverage Problem")
    
    model.Elements = pyo.Set(doc="collects elements to cover")
    model.Sets = pyo.Set(doc="collects sets to choose from")
    model.FeasibleSets = pyo.Set(model.Elements, doc="collects the sets that can be selected to cover element k")
    
    model.max_num_sets = pyo.Param(mutable=True)
    
    model.element_is_covered = pyo.Var(model.Elements, within=pyo.Binary)
    model.set_is_selected = pyo.Var(model.Sets, within=pyo.Binary)
    
    @model.Constraint(model.Elements, doc="coverage constraint")
    def coverage_indicator_constraint(block, k):
        return sum(block.set_is_selected[j] for j in block.FeasibleSets[k]) >= block.element_is_covered[k]
    
    @model.Constraint(doc="select at most K sets")
    def limit_number_of_sets(block):
        return sum(block.set_is_selected[j] for j in block.Sets) <= block.max_num_sets
    
    @model.Objective(doc="Maximize coverage", sense=pyo.maximize)
    def coverage(block):
        return pyo.summation(block.element_is_covered)
    
    return model

