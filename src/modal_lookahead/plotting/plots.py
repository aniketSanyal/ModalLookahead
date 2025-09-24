import numpy as np, matplotlib.pyplot as plt
from .style import set_style, colors20, markers

def plot_distance_curves(curves, title, outfile_base):
    set_style(); alpha=0.9; lw=3.0; mew=0.7; ms=2
    palette=[colors20[8], colors20[0], colors20[2], colors20[14]]
    mseq=[markers[3], markers[6], markers[-1], markers[0]]
    mark=[5, 7, 9, 11]
    fig,ax=plt.subplots(figsize=(13,6)); it=np.arange(len(curves[0][1]))
    for i,(name,y) in enumerate(curves):
        ax.plot(it,y,lw=lw,color=palette[i%len(palette)],label=name,alpha=alpha,
                marker=mseq[i%len(mseq)],markersize=ms,markeredgewidth=mew,markeredgecolor='k',markevery=mark[i%len(mark)])
    ax.tick_params(labelsize='large',width=3); ax.set_xlabel('Iteration',fontsize=22)
    ax.set_ylabel("Distance to equilibrium  ||[x;y]||_2",fontsize=22)
    ax.margins(x=0,y=0); ax.ticklabel_format(useOffset=False,useMathText=True,scilimits=(-5,1),style="scientific")
    ax.tick_params(labelsize=16); ax.grid(alpha=0.3); ax.legend(fontsize=14,frameon=False,loc='upper right',labelspacing=0.12)
    ax.xaxis.get_offset_text().set_fontsize(16); ax.set_title(title,fontsize=20,pad=10); plt.tight_layout()
    for ext,dpi in [("png",800),("pdf",None),("svg",None)]:
        p=f"{outfile_base}.{ext}"; (fig.savefig(p,bbox_inches='tight',dpi=dpi) if dpi else fig.savefig(p,bbox_inches='tight'))
    plt.close(fig)
