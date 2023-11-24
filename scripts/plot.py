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
import re

import yaml

import pyomo.environ as pyo
from pyomo.opt import SolverResults
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd


HOME = Path(__file__).resolve().parent
DATA_DIR = HOME.parent / 'data'
OUT_DIR = HOME.parent / 'figs'

mystyle = dict(
    facecolors="white",
    edgecolor="black",
    linewidth=1,
    marker='D',
    s=15
)


def _set_log_limits(ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    xmin, xmax = ax.get_xlim()
    xmin = np.power(10,np.floor(np.log10(xmin)))
    xmax = np.power(10,np.ceil(np.log10(xmax)))

    ymin, ymax = ax.get_ylim()
    ymin = np.power(10,np.floor(np.log10(ymin)))
    ymax = np.power(10,np.ceil(np.log10(ymax)))

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)


def _draw_feas_region(series: pd.Series, ax=None, **kwargs):
    """draws the feasable region as detemined by VCI PiC-LF03 Appendix A
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        
    ax.plot(
        [series["left x1 [m3/h]"],series["left x2 [m3/h]"],series["right x2 [m3/h]"],series["right x1 [m3/h]"],series["left x1 [m3/h]"]],
        [series["left y1 [mFs]"],series["left y2 [mFs]"] ,series["right y2 [mFs]"],series["right y1 [mFs]"],series["left y1 [mFs]"]],
        **kwargs
    )

def _draw_reverse_engineered_region(frame: pd.DataFrame, select: str, ax=None, **kwargs):
    """draws the feasable region as detemined by VCI PiC-LF03 Appendix A
    reversed engineered version
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    vals = frame.loc[:,"nominal head [mFs]"].unique()
    vals = np.append(vals, 3)
    vals.sort()
    head_values = vals.tolist()

    vals = frame.loc[:,"nominal head [mFs]"].unique()
    vals = np.append(vals, 3)
    vals.sort()
    head_values = vals.tolist()
    
    pump = frame.loc[select]
    
    Q_iso = pump["nominal flow rate [m3/h]"]
    H_iso = pump["nominal head [mFs]"]
    
    idx = head_values.index(H_iso)-1
    H_iso_smaller = head_values[idx]
    
    
    k1 = pump["F1"]
    k2 = pump["F2"]
    k3 = pump["F3"]
    k4 = pump["F4"]
    
          
    coords = np.array([
        [k3*Q_iso, k1*H_iso], # upper left
        [k4*Q_iso, H_iso], # upper right
        [Q_iso, H_iso_smaller], # lower right
        [k3*Q_iso, k2*H_iso_smaller], # lower left
        [k3*Q_iso, k1*H_iso] # upper left
    ])

    ax.plot(coords[:,0], coords[:,1], **kwargs)


def draw_ranges(frame=None, ax=None, fname: str="2020-08-05-lf03-anhang-a-berechnungsblatt.csv", **kwargs):
   
    if frame is None:
        frame = pd.read_csv(
            DATA_DIR / fname,
            sep=",",
            index_col="register"
        )

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    for idx in range(frame.shape[0]):
        _draw_feas_region(frame.iloc[idx], ax=ax, **kwargs)


def loglog_scatter_plot(frame, x, y, xlabel=None, ylabel=None, ax=None, **kwargs):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    x_data = frame.loc[:,x] if isinstance(x,str) else frame.iloc[:,x]
    y_data = frame.loc[:,y] if isinstance(x,str) else frame.iloc[:,y]
    
    ax.scatter(x_data, y_data,**kwargs)

    ax.loglog()
    _set_log_limits(ax)

    ax.set_aspect('equal')
    ax.grid(which="major")

    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)


def gen_fig_ranges_overview(fname: str="2020-08-05-lf03-anhang-a-berechnungsblatt.csv"):
    """Generates figure 2 from article

    Inputs:
    ------
        fname (str): filename of the table containing VCI Limits
    """
    frame = pd.read_csv(
        DATA_DIR / fname,
        sep=",",
        index_col="register"
    )

    scatterstyle = dict(
        facecolors="black",
        edgecolor="black",
        linewidth=1,
        marker='o',
        s=15,
    )

    linesstyle = dict(linestyle='-',color='k')
    
    fig,axs = plt.subplots(1,2,sharey=True,figsize=(12,6))
    
    for speed, ax in zip([1450,2900],axs):
        
        df = frame.where(frame["speed [min-1]"] == speed)
        loglog_scatter_plot(df, ax=ax, x="nominal flow rate [m3/h]",y="nominal head [mFs]",label="nominal duty point", **scatterstyle)
        draw_ranges(frame=df, ax=ax, **linesstyle)
       
        ax.loglog()
        
        ax.set_xlim(1e-1,1e3)
        ax.set_ylim(1e0,1e3)

        _ = ax.set_xlabel(r"FLOW RATE $Q$ in $\mathrm{m}^3/\mathrm{h}$")
        _ = ax.set_ylabel(r"HEAD $H$ in $\mathrm{m}$")
        _ = ax.set_title(r"SPEED {} ".format(speed) + r"$\mathrm{min}^{-1}$")
    
    return fig


def gen_fig_sample(
        fname_elements: str,
        fname_pumps: str="2020-08-05-lf03-anhang-a-berechnungsblatt.csv"
    ):

    sets = pd.read_csv(
            DATA_DIR / fname_pumps,
            sep=",",
            index_col="register"
        )
    sample = pd.read_csv(fname_elements, index_col="element_id")

    elements = sample[sample["mask_total"]]
    rejected = sample[~sample["mask_total"]]

    linesstyle = dict(linestyle='-', linewidth=.5, color='gray')

    fig = plt.figure()
    ax = fig.gca()

    draw_ranges(frame=sets, ax=ax, **linesstyle)

    ax.scatter(
        elements["flow_rate"],
        elements["head"],
        facecolors="white",
        edgecolor="black",
        linewidth=1,
        marker='D',
        s=15,
        label=r"accepted part of the sample $N = {}$".format(elements.shape[0])
    )

    ax.scatter(
        rejected["flow_rate"],
        rejected["head"],
        facecolors="white",
        edgecolor="gray",
        linewidth=1,
        marker='D',
        s=15,
        label=r"rejected part of the sample $\overline{N}$" + "= {}".format(rejected.shape[0])
    )

    # limits
    xdata = sample["flow_rate"]
    ydata = sample["head"]

    xmin, xmax = np.power(10,np.floor(np.log10(xdata).min())), np.power(10,np.ceil(np.log10(xdata).max()))
    ymin, ymax = np.power(10,np.floor(np.log10(ydata).min())), np.power(10,np.ceil(np.log10(ydata).max()))
    
    _ = ax.loglog()
    _ = ax.grid(which="major")
    _ = ax.set_aspect("equal")
    
    _ = ax.set_xlabel("FLOW RATE $Q$ in $\mathrm{m}^3/\mathrm{h}$")
    _ = ax.set_ylabel("HEAD $H$ in $\mathrm{m}$")
    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc="lower left", edgecolor='none')

    return fig


def get_covered(results: SolverResults):
    prog = re.compile("^element_is_covered\[(\d+)\]$")
    for var in results.Solution[0].Variable.keys():
        m = prog.match(var)
        if m: 
            idx = int(m.group(1))
            yield idx
        else:
            continue


def get_selected(results: SolverResults):
    prog = re.compile("^set_is_selected\[(\w+)\]$")
    for var in results.Solution[0].Variable.keys():
        m = prog.match(var)
        if m: 
            idx = m.group(1)
            yield idx
        else:
            continue


def gen_fig_solution(
    fname_sol: str=None,
    model: pyo.ConcreteModel=None,
    fname_elements: str=None,
    elements: pd.DataFrame=None,
    fname_sets: str="2020-08-05-lf03-anhang-a-berechnungsblatt.csv",
    sets: pd.DataFrame=None,
    ax=None
):
    if fname_sol:
        results = SolverResults()
        results.read(filename=fname_sol)

        coverage = results.Solution[0].Objective["coverage"]["Value"]
        idx_covered = list(get_covered(results))
        idx_selected = list(get_selected(results))
        max_num_sets = len(idx_selected)
    elif model:
        coverage = pyo.value(model.coverage)
        idx_covered = (k for k in model.Elements if abs(pyo.value(model.element_is_covered[k]) - 1.0) <= 1e-4)
        idx_selected = (k for k in model.Sets if abs(pyo.value(model.set_is_selected[k]) - 1.0) <= 1e-4)
        max_num_sets = pyo.value(model.max_num_sets)
    else:
        raise IOError("either model or solution file needed")


    if sets is None:
        sets = pd.read_csv(
            DATA_DIR / fname_sets,
            sep=",",
            index_col="register"
        )

    if elements is None:
        sample = pd.read_csv(fname_elements, index_col="element_id")
        elements = sample[sample["mask_total"]]


    fig = plt.figure()
    ax = fig.gca()
    
    loglog_scatter_plot(
        elements.loc[idx_covered],"flow_rate","head",ax=ax,
        facecolors="black",
        edgecolor="black",
        linewidth=1,
        marker='D',
        s=8,
        label="covered"
    )

    loglog_scatter_plot(
        elements[~elements.index.isin(idx_covered)],"flow_rate","head",ax=ax,
        facecolors="white",
        edgecolor="black",
        linewidth=1,
        marker='D',
        s=8,
        label="not covered"
    )

    draw_ranges(frame=sets.loc[idx_selected], ax=ax, color='black', linestyle='-', linewidth=1.5, marker='None', label="selected")
    draw_ranges(frame=sets.loc[~sets.index.isin(idx_selected)], ax=ax, color='gray', linestyle='-', linewidth=0.5, marker='None', label="not selected")

    _ = ax.set_xlabel("FLOW RATE $Q$ in $\mathrm{m}^3/\mathrm{h}$")
    _ = ax.set_ylabel("HEAD $H$ in $\mathrm{m}$")

    rel_cov = coverage / elements.shape[0]
    _ = ax.set_title("RELATIVE COVERAGE {:.4f}, N = {}".format(rel_cov, max_num_sets))

    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc="upper left", frameon=False)

    ax.set_xlim([.1,1000])
    ax.set_ylim([1,1000])
    
    ax.set_aspect('equal')
    ax.grid(which="major")
    return fig


def gen_pareto_plot(frame: pd.DataFrame=None, fname=None):

    if frame is None:
        df = pd.read_csv(fname)

    xupper = np.ceil(df.shape[0]/10)*10 if df.shape[0] <= 65 else 68

    fig = plt.figure()
    ax1 = fig.gca()
    ax2 = ax1.twinx()

    min_cover = df["max_num_sets"].max()
    ax1.plot([min_cover,min_cover], [0,1], linewidth=1, color="black")
    ax1.scatter(
        df["max_num_sets"],
        df["rel_coverage"],
        facecolors="black",
        edgecolor="black",
        linewidth=1,
        marker='D',
        s=10
    )

    ax2.plot([0,xupper], [.01,.01], linewidth=.5, color="gray",zorder=0)
    ax2.scatter(
        df["max_num_sets"],
        df["rel_coverage"].diff(),
        facecolors="white",
        edgecolor="black",
        linewidth=1,
        marker='D',
        s=12
    )

    
    ax1.set_xlabel(r"NUMBER OF ISO-SIZES in PUMP POOL $K$")
    ax1.set_ylabel(r"RELATIVE COVERAGE OF APPLICATIONS $z$")
    ax2.set_ylabel(r"MARGINAL GAIN per ADDITIONAL ISO-SIZE $\Delta z / \Delta K$")

    ax1.set_xlim(0,xupper)
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,.1)
    
    ax1.spines[['right', 'top']].set_visible(False)
    ax2.spines[['left', 'top']].set_visible(False)

    return fig


if __name__ == "__main__":

    parser = ArgumentParser(description="plotting utility")

    parser.add_argument("-s", "--save", dest="SAVE", action=BooleanOptionalAction, required=False, default=False)
    parser.add_argument("-d", "--dir", dest="OUT_DIR", type=Path, required=False, default=HOME.parent/"figs")
    parser.add_argument("-f", "--format", dest="FORMAT", type=str, required=False, default="png")
    parser.add_argument("-c", "--config", dest="CONFIG", type=Path, required=True)
    
    args = parser.parse_args()

    with open(args.CONFIG,"r") as f:
        config = yaml.safe_load(f)

    figs = dict()
    figs["ranges_overview"] = gen_fig_ranges_overview(fname=config["ranges_overview"])
    figs["sample"] = gen_fig_sample(fname_elements=config["sample"]["fname_elements"])
    figs["solution"] = gen_fig_solution(
            fname_sol=config["solution"]["fname_sol"],
            fname_elements=config["solution"]["fname_elements"]
        )
    figs["pareto"] = gen_pareto_plot(fname=config["pareto"])

    if args.SAVE:
        if not args.OUT_DIR.is_dir():
            args.OUT_DIR.mkdir()

        for key, fig in figs.items():
            fname = args.OUT_DIR / "figure_{0}_{1}".format(
                datetime.today().strftime('%y%m%d'), \
                key
            )
            rcParams["savefig.format"] = args.FORMAT
            fig.savefig(fname)
            