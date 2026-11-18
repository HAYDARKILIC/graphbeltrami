#!/usr/bin/env python3
"""Regenerate the five figures of the paper from the raw results.

Usage:
    python scripts/make_figures.py results/ figures/

Figure_1 and Figure_2 appear in the main text, Figure_S1 to Figure_S3 in the
Supplementary Material. The discretisation figure (Figure_2) is built from the
corrected-step-rule pass MAIN_e296b125, never from results/superseded/.
"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
W = 5.4  # inches = 390pt text width
plt.rcParams.update({"savefig.dpi":300,"savefig.bbox":"tight","font.family":"serif","font.size":9,
 "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
 "axes.grid":True,"grid.alpha":0.25,"axes.spines.top":False,"axes.spines.right":False,
 "lines.linewidth":1.5,"lines.markersize":4,"pdf.fonttype":42})
OK = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00","purple":"#CC79A7","sky":"#56B4E9","black":"#000000","yellow":"#F0E442"}
NAME = {"learned_node_gauge":r"Aux-$\psi$","learned_edge_gauge":r"Aux-$\psi a^{+}$","lorentzian":r"Aux-$\psi a^{\pm}$",
        "blend_induced":"Induced metric","gprgnn":"GPR-GNN","fagcn":"FAGCN","gcn":"GCN","mlp":"MLP","grand":"GRAND","graff":"GRAFF"}
VC = {"learned_node_gauge":OK["red"],"learned_edge_gauge":OK["purple"],"lorentzian":OK["blue"]}
def save(f,n):
    for e in ("pdf","png"): f.savefig(f"{OUT}/{n}.{e}")
    plt.close(f)
def log2x(ax, ticks):
    ax.set_xscale("log",base=2); ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks])); ax.xaxis.set_minor_locator(NullLocator())
def logy(ax, ticks):
    ax.set_yscale("log"); ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks])); ax.yaxis.set_minor_locator(NullLocator())

# ---------------- Figure 1: learned responses
E2 = json.load(open(f"{R}/e2_response_MAIN_ae15dbc3.json"))
fig, axs = plt.subplots(2,3,figsize=(W,3.45))
for i,(ds,e) in enumerate(E2.items()):
    ax=axs[i//3,i%3]
    spec=next((e[v]["spectrum"] for v in VC if e.get(v,{}).get("spectrum")),None)
    if spec:
        ax2=ax.twinx(); ax2.hist(spec,bins=50,color="0.6",alpha=0.3); ax2.set_yticks([]); ax2.grid(False)
        for s_ in ax2.spines.values(): s_.set_visible(False)
        ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    for v,ls in (("learned_node_gauge","-"),("learned_edge_gauge",(0,(4,1.5))),("lorentzian","-")):
        ax.plot(e[v]["grid"],e[v]["h"],color=VC[v],ls=ls,label=NAME[v],lw=1.6 if v!="lorentzian" else 1.8)
    mu=np.array(e["gprgnn"]["mu"]); h=np.array(e["gprgnn"]["h"]); lam=1-mu; k=(lam>=0)&(lam<=2)
    ax.plot(lam[k],h[k]/max(abs(h[k]).max(),1e-12),ls=(0,(1.5,1.2)),color=OK["black"],lw=1.4,label="GPR-GNN")
    ax.axhline(1,color="0.5",lw=0.6,ls=":")
    ax.set_title(ds,pad=3)
    if i>=3: ax.set_xlabel(r"graph frequency $\lambda$")
    if i%3==0: ax.set_ylabel(r"response $h(\lambda)$")
h_,l_=axs[0,0].get_legend_handles_labels()
fig.legend(h_,l_,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(rect=(0,0.05,1,1),h_pad=0.6,w_pad=0.4); save(fig,"Figure_1")

# ---------------- Figure 2: discretisation, per variant
e3=pd.read_csv(f"{R}/e3_solver_MAIN_e296b125.csv")
fig, axs = plt.subplots(2,3,figsize=(W,3.45))
Ts=[1,2,4,8,16,32]
for r,sc in enumerate([False,True]):
    for m in ("learned_node_gauge","lorentzian"):
        for prop,ls,mk in (("exact","-","o"),("euler",(0,(4,1.5)),"s")):
            g=e3[(e3.scale_operator==sc)&(e3.model==m)&(e3.propagator==prop)].groupby("T").mean(numeric_only=True)
            c=VC[m]; mf = c if prop=="exact" else "white"
            lab=f"{NAME[m]}, {'exact' if prop=='exact' else 'Euler'}"
            axs[r,0].plot(g.index,g.delivered_T,ls=ls,marker=mk,color=c,mfc=mf,label=lab)
            axs[r,1].plot(g.index,g.n_matvec,ls=ls,marker=mk,color=c,mfc=mf)
            axs[r,2].plot(g.index,100*g.test_primary,ls=ls,marker=mk,color=c,mfc=mf)
    axs[r,0].plot([1,32],[1,32],":",color="0.4",lw=0.9)
    axs[r,1].axhline(72,color="0.5",lw=0.7)
    axs[r,0].set_ylabel(("Un-normalised" if not sc else "Normalised")+"\ndelivered horizon")
    axs[r,1].set_ylabel("products / pass"); axs[r,2].set_ylabel("primary metric (%)")
    for a in axs[r]: log2x(a,Ts)
    axs[r,0].set_yscale("log",base=2); axs[r,0].yaxis.set_major_locator(FixedLocator([0.25,1,4,16]))
    axs[r,0].yaxis.set_major_formatter(FixedFormatter(["0.25","1","4","16"])); axs[r,0].yaxis.set_minor_locator(NullLocator())
    logy(axs[r,1],[2,8,32,128])
for a in axs[1]: a.set_xlabel("requested $T$")
axs[0,0].set_title("(a) horizon",pad=3); axs[0,1].set_title("(b) cost",pad=3); axs[0,2].set_title("(c) accuracy",pad=3)
h_,l_=axs[0,0].get_legend_handles_labels()
fig.legend(h_,l_,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.04),columnspacing=1.0,handlelength=2.2)
fig.tight_layout(rect=(0,0.06,1,1),h_pad=0.6,w_pad=0.5); save(fig,"Figure_2")

# ---------------- Figure S1: convergence (worst case)
e1=pd.read_csv(f"{R}/e1_convergence_MAIN_ae15dbc3.csv")
fig, axs = plt.subplots(1,2,figsize=(W,2.3))
c=e1[e1.knob=="cheb_order"].groupby(["T","value"]).rel_err.max()
for T,col,mk in ((1.0,OK["sky"],"o"),(4.0,OK["red"],"s"),(16.0,OK["blue"],"^")):
    s=c.loc[T]; axs[0].semilogy(s.index,s.values,marker=mk,color=col,label=f"$T={T:g}$")
g=e1[e1.knob=="cg_iters"].groupby(["dataset","value"]).rel_err.max()
for ds,col,mk in (("Cora",OK["green"],"o"),("Chameleon-f",OK["orange"],"s"),("Squirrel-f",OK["purple"],"^")):
    s=g.loc[ds]; axs[1].semilogy(s.index,s.values,marker=mk,color=col,label=ds)
for a in axs:
    a.axhline(2.2e-16,color="0.3",ls=":",lw=0.9,label="float64 round-off")
    a.set_ylim(1e-16,1e-3)
axs[0].set_xlabel("Chebyshev order $K$"); axs[1].set_xlabel("CG iterations $J$")
axs[0].set_ylabel("worst relative error")
axs[0].legend(frameon=False,title="max over graphs",title_fontsize=7.5); axs[1].legend(frameon=False,title="max over $T$",title_fontsize=7.5)
fig.tight_layout(w_pad=1.2); save(fig,"Figure_S1")

# ---------------- Figure S2: stability per dataset
e4=pd.read_csv(f"{R}/e4_stability_MAIN_ae15dbc3.csv")
fig, axs = plt.subplots(2,2,figsize=(W,3.7),sharex=True)
for r,ds in enumerate(["Chameleon-f","Squirrel-f"]):
    for m in ("learned_node_gauge","learned_edge_gauge","lorentzian"):
        for mode,ls,mk in (("shift","-","o"),("free",(0,(4,1.5)),"s")):
            g=e4[(e4.dataset==ds)&(e4.model==m)&(e4.stability_mode==mode)].sort_values("T")
            lab=f"{NAME[m]}, {'with shift' if mode=='shift' else 'without shift'}"
            axs[r,0].plot(g["T"],g["margin"],ls=ls,marker=mk,color=VC[m],mfc=VC[m] if mode=="shift" else "white",label=lab)
            axs[r,1].plot(g["T"],100*g["test_primary"],ls=ls,marker=mk,color=VC[m],mfc=VC[m] if mode=="shift" else "white")
    axs[r,0].axhline(0,color="0.3",ls=":",lw=0.9)
    axs[r,0].set_ylabel(f"{ds}\n"+r"margin $\beta-|\lambda_{\min}|$"); axs[r,1].set_ylabel("accuracy (%)")
    for a in axs[r]: log2x(a,[1,4,16])
for a in axs[1]: a.set_xlabel("horizon $T$")
h_,l_=axs[0,0].get_legend_handles_labels()
fig.legend(h_,l_,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(rect=(0,0.09,1,1)); save(fig,"Figure_S2")

# ---------------- Figure S3: mean ranks with CD
st=json.load(open(f"{R}/statistics_MAIN_ae15dbc3.json"))["heterophilic"]
mr=pd.Series(st["mean_ranks"]).sort_values(); cd=st["cd"]
fig, ax = plt.subplots(figsize=(W,2.9))
y=np.arange(len(mr))[::-1]
cols=[VC.get(m,"0.35") for m in mr.index]
ax.hlines(y,1,mr.values,color="0.85",lw=1)
ax.scatter(mr.values,y,color=cols,zorder=3,s=28)
ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in mr.index])
best=mr.values[0]
ax.axvspan(best,min(best+cd,10.2),color=OK["sky"],alpha=0.15,lw=0)
ax.annotate("",xy=(best+cd,len(mr)-0.35),xytext=(best,len(mr)-0.35),arrowprops=dict(arrowstyle="<->",color="0.2",lw=1))
ax.text(best+cd/2,len(mr)-0.25,f"critical difference = {cd:.2f}",ha="center",va="bottom",fontsize=8)
ax.set_xlim(1,10.2); ax.set_ylim(-0.7,len(mr)+0.4)
ax.set_xlabel("mean rank over the five heterophilic graphs (lower is better)")
ax.grid(axis="y",alpha=0)
fig.tight_layout(); save(fig,"Figure_S3")
