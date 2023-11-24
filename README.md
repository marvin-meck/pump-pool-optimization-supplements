# A Set-Covering Approach to Minimize the Variety of Standard Chemical Process Pumps in Equipment Pools -- Supplementary material

*Abstract*: A decision framework is presented to minimize the number of distinct ISO 2858 compliant standard chemical process pumps in an equipment pool. 
Two model variants are introduced: (i) the set cover problem to find the minimum number of ISO-sizes required to meet a given set of specifications, and (ii) the maximum coverage problem to allow for a trade-off to be made between application coverage and equipment variety in the pump pool. 
The decision-models are applied to a data set containing specifications for pumping applications provided by Evonik Operations GmbH. 
A kernel density estimator is trained using that data to provide a non-parametric generative model that is used to anticipate likely future specifications for which a pump pool is to be set up. 
The decision framework is applied for a generated sample of 3268 applications choosing from 68 available ISO sizes. 
The results suggest, that equipment pools comprising just a subset of all available ISO-sizes may be sufficient to cover a broad range of applications. 
Specifically, in our case study 36 out of 68 standard pumps cover 95% of all applications suitable for standard chemical process pumps; which is 62% of the total specifications sample generated. 
Numerically, we found the model to scale well with overall problem size. 
The entire data processing pipeline including all pre-processing of data can be executed in the order of a few seconds on standard laptop computer hardware, making the approach computational tractable even for large scale industrial analyses. 
By including an established model of the centrifugal pump operating ranges taken from a dimensioning tool developed at German chemical industry association VCI, we eliminate the need for further specialized software and minimize equipment-data requirements. 
Thus, we provide a simple, clear and concise model that equipment managers can readily integrate in their workflows.

## About

This repository contains supplementary material to a research article titled *A Set-Covering Approach to Minimize the Variety of Standard Chemical Process Pumps in Equipment Pools*. 
The supplementary material contained in this repository comprises the following:

1. A Jupyter Notebook `note_230808_PumpPoolOptimization_meck.ipynb` intended to document the data processing pipeline.

2. *Pyomo* [[1]](#references) implementations of both the *Set Cover Problem* (`model_set_cover.py`) and the *Maximum Coverage Problem* (`model_maximum_coverage.py`)

3. Example *Data Command* files to be used with the model implementations mentioned under 2.

4. The `./data` directory intended to hold (raw) input data. 
    For instance, to reproduce the results reported in the article, download the *Microsoft Excel Dimensioning Tool* (VCI PiC-LF03 Appendix A [[2]](#references)) `2020-08-05-lf03-anhang-a-berechnungsblatt.xls` from the [VCI website](https://www.vci.de/services/leitfaeden/kreiselpumpenaggregate-lf03-anwendung-bewertung-strategien-gemaess-din-en-iso-2858-vci-leitfaden.jsp), and copy the file here (see below).

5. The `./scripts` directory containing four Python modules that make up the entire data processing pipeline. 
    1.  `init.py` to extract the operating limits defined by PiC-LF03 Appendix A into a `.csv` file. 
        To do so, download `2020-08-05-lf03-anhang-a-berechnungsblatt.xls` from the [VCI website](https://www.vci.de/services/leitfaeden/kreiselpumpenaggregate-lf03-anwendung-bewertung-strategien-gemaess-din-en-iso-2858-vci-leitfaden.jsp). 
        Copy the file to `./data/` and run `init.py` as
        ```shell
        python "./scripts/init" -i data/2020-08-05-lf03-anhang-a-berechnungsblatt.xls
        ```
        **Note**: `2020-08-05-lf03-anhang-a-berechnungsblatt.csv` is required to generate most of the plots and---more importantly---the *Data Command* files to reproduce (or generate new)results.  
    2.  `sample.py` to create a KDE and generate a new sample. This is meant to be run as a command line app. 
        The idea is, you provide a table with columns `flow_rate`, `head` and `kin_viscosity` (given in m3/h, m, and cSt respectively). This can be via a file or database query. 
        From that `sample.py` generates a sample for you. 
        Type the following for more info on how to use it. 
        ```shell
        python sample.py --help
        ```  
    3.  `preprocess.py` generates the *Data Command* file for the *Set Cover Problem*. 
        Meant to be run from the command line. 
        Type the following for more info on how to use it. 
         ```shell
        python preprocess.py --help
        ```  
    4.  `pareto.py` is a utility script to generate a Pareto frontier. 
        It first solves the *Set Cover Problem* to bound the parameter $K$ in the cardinality constraint above ($\bar{K}$). 
        Thereafter, the *Maximium Coverage Problem* is solved for every integer value $0 < K < \bar{K}$. 
         ```shell
        python pareto.py --help
        ```  
    5.  `plot.py` contains plotting utilities. Run `plot.py` from the command line or use as an import module to generate your own plots. For more:
        ```shell
        python plot.py --help
        ```
6. The `./results` directory containing all data generated as part of the analysis. 
    This includes  
    1.  Application data sampled from the Kernel Density Estimator (KDE)  
    2.  [*Pyomo Data Command*](https://pyomo.readthedocs.io/en/stable/working_abstractmodels/data/datfiles.html#the-param-command) files generated for that sample.  
    3.  Optimization results in structured [(YAML)](https://yaml.org/) generated within `pyomo`.  
    4.  Numerical values for the Pareto frontier  
    5.  Results from the computational study


## How can I reproduce the results? 

Please read the notebook `note_230808_PumpPoolOptimization_meck.ipynb` for instructions on how to reproduce the results or create your own. 

## How can I recover a solution from a `.yml` results file?

```python
from pyomo.opt import SolverResults
from model_maximum_coverage import pyomo_create_model

if __name__ == "__main__":
    model = pyomo_create_model()
    instance = model.create_instance(data="example.dat")

    results = SolverResults()
    results.read(filename="results.yml")

    # fix the solution object, otherwise results.solutions.load_from(...) won't work
    results.solution(0)._cuid = False
    results.solution.Constraint = {}

    instance.solutions.load_from(results)
    
    # default_variable_value=0 doesn't work because smap_id = None, 
    # so we set them manually
    for var in instance.component_data_objects(pyo.Var):
        if var.value is None:
            var.value = 0

    # test output
    instance.set_is_selected.pprint()
```

## Third party dependencies

Applications within this source tree depend on several third party software modules and materials (the "dependencies") distributed primarily under permissive software licenses. 
Use of these dependencies is subject to the terms and conditions of their respective licenses.
We included a list of primary dependencies and verbatim copies of the licenses/ copyright notices and terms of use provided with those dependencies. 

| Dependency    | Version   | License                                                   |
|---------------|-----------|-----------------------------------------------------------|
| ipywidgets    | 8.1.1     | [BSD 3-Clause](license_dependencies/IPYWIDGETS_LICENSE)   |
| matplotlib    | 3.8.2     | [BSD-style](license_dependencies/MATPLOTLIB_LICENSE)      |
| notebook      | 7.0.6     | [BSD 3-Clause](license_dependencies/NOTEBOOK_LICENSE)     |
| pandas        | 2.1.3     | [BSD 3-Clause](license_dependencies/PANDAS_LICENSE)       |
| pyodbc        | 5.0.1     | [MIT-0](license_dependencies/PYODBC_LICENSE)              |
| Pyomo         | 6.6.2     | [BSD-style](license_dependencies/PYOMO_LICENSE)           |
| scikit-learn  | 1.3.2     | [BSD 3-Clause](license_dependencies/SCIKIT-LEARN_COPYING) |
| uncertainties | 3.1.7     | [BSD-style](license_dependencies/UNCERTAINTIES_LICENSE)   |


In order to reproduce the results reported in our paper (or to generate new results for another specifications data
set), additionally the *Microsoft Excel Dimensioning Tool* [[2]](#references)  distributed with the *PiC LF03 Guideline on centrifugal pump units* [[3]](#references) developed by German chemical industry association VCI is required for data extraction. 
The *PiC LF03 Guideline on centrifugal pump units* and *Microsoft Excel Dimensioning Tool* fall under the terms of use provided with those resources; the copyright lies with VCI (see [copyright and terms of use (German)](license_dependencies/VCI_PiC_LF03_APPENDIX_A)).  


## References

[1]     Hart, William E., William E. Hart; Watson, Jean-Paul; Laird, Carl D.; Nicholson, Bethany L.; Siirola, John D. (2017): 
        Pyomo. Optimization Modeling in Python. Second edition /  William E Hart [and six others].
        Cham: Springer (Springer Optimization and Its Applications, 67).

[2]     Verband der Chemischen Industrie e.V. (2020): Excel calculation sheet for determining the size by DIN EN ISO 2858 size maps; Excel-Berechnungsblatt zur Ermittlung der Baugröße durch DIN EN ISO 2858 Baugrößenkennfelder (original title). Verband der Chemischen Industrie e.V. (VCI). 
        Available online at https://www.vci.de/services/leitfaeden/kreiselpumpenaggregate-lf03-anwendung-bewertung-strategien-gemaess-din-en-iso-2858-vci-leitfaden.jsp.  

[3]     Verband der Chemischen Industrie e.V. (2020): VCI-guide -- Centrifugal pump units: Application, assessment, strategies in accordance
with DIN EN ISO 2858; VCI-Leitfaden -- Kreiselpumpenaggregate. Anwendungen, Bewertung, Strategien gemäß DIN EN ISO 2858 (original title). Verband der Chemischen Industrie e.V. (VCI). Available online at https://www.vci.de/services/leitfaeden/kreiselpumpenaggregate-lf03-anwendung-bewertung-strategien-gemaess-din-en-iso-2858-vci-leitfaden.jsp.  


## Funding

We thank the German Federal Ministry for Economic Affairs and Climate Action (BMWK)
for financially supporting the work within the scope of the HECTOR research project 
as part of the ENPRO 2.0 initiative (funding code: 03EN2006A).
