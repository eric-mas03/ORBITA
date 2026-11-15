"""Head/horizon curves and variable-level uncertainty plots from frozen CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def head_plot(cells, metric, destination, title):
    heads = sorted(cells["head"].unique())
    groups = cells.groupby(["method","head","horizon_hours"])[metric].mean().reset_index()
    methods = sorted(groups.method.unique())
    fig,axes = plt.subplots(int(np.ceil(len(heads)/2)),2,figsize=(13,3*int(np.ceil(len(heads)/2))),squeeze=False)
    colors = {method:plt.get_cmap("tab20")(i) for i,method in enumerate(methods)}
    for ax,head in zip(axes.flat,heads):
        for method in methods:
            sub = groups[groups.method.eq(method) & groups["head"].eq(head)].sort_values("horizon_hours")
            ax.plot(sub.horizon_hours,sub[metric],"o-",label=method,color=colors[method],markersize=3,linewidth=1)
        ax.set_title(head)
        ax.set_xticks([1,4,8,24])
        ax.set_xlabel("Horizonte (h)")
        ax.set_ylabel("Pinball normalizada ↓")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(heads):]:
        ax.set_visible(False)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.975),ncol=3,fontsize=7)
    fig.suptitle(title,y=.998)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(destination,dpi=150)
    plt.close(fig)


def main():
    root = ROOT / "outputs/model_campaign_v2"
    dev = pd.read_csv(root / "development/dev_by_variable_horizon.csv")
    head_plot(dev,"normalized_pinball",root / "development/dev_head_comparison.png",
              "V2 · desarrollo · comparación por cabeza y horizonte")
    out = root / "workspace/final_evaluation"
    cells = pd.read_csv(out / "metrics_by_variable_horizon.csv")
    head_plot(cells,"pinball_standardized",out / "final_head_comparison.png",
              "V2 · prueba final · curvas descriptivas; conservar soporte por variable")
    ensemble = cells[cells.method.eq("ensemble_calibrated")]
    fig,axes = plt.subplots(1,3,figsize=(14,12))
    for ax,column,title,cmap in zip(axes,["pinball_standardized","coverage80_empirical","episodes"],
        ["Pinball normalizada ↓","Cobertura empírica de la banda","Episodios con etiqueta real"],["YlOrRd","RdYlBu","Blues"]):
        grid = ensemble.pivot(index="variable",columns="horizon_hours",values=column).reindex(columns=[1,4,8,24])
        kwargs = dict(vmin=0,vmax=1) if column == "coverage80_empirical" else {}
        im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()),aspect="auto",cmap=cmap,**kwargs)
        ax.set_yticks(range(len(grid)),grid.index,fontsize=8)
        ax.set_xticks(range(4),["1 h","4 h","8 h","24 h"])
        ax.set_title(title,fontsize=11)
        fig.colorbar(im,ax=ax,shrink=.7)
        if column == "episodes":
            rows,cols = np.nonzero(grid.to_numpy()<10)
            for row,col in zip(rows,cols):
                ax.text(col,row,"×",ha="center",va="center",color="#9a2e20",fontsize=12)
    fig.suptitle("Ensemble ajustado · prueba final · × = menos de diez episodios; no ocultar celdas escasas")
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out / "ensemble_variable_horizon_evaluation.png",dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
