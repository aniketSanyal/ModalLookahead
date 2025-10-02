import numpy as np, matplotlib.pyplot as plt
from .style import set_style, colors20, markers

def plot_distance_curves(curves, title, outfile_base):
    set_style(); alpha=0.9; lw=3.0; mew=0.7; ms=9
    palette=[colors20[8], colors20[0], colors20[2], colors20[14],colors20[4], colors20[6], colors20[10], colors20[12]]
    mseq=[markers[3], markers[6], markers[-1], markers[0],  markers[4], markers[2], markers[5], markers[1]]
    mark=[100, 150, 200, 250, 300, 350]
    fig,ax=plt.subplots(figsize=(13,6)); 
    it=np.arange(len(curves[0][1]))
    for i,(name,y) in enumerate(curves):
        ax.plot(it,y,lw=lw,color=palette[i%len(palette)],label=name,alpha=alpha,
                marker=mseq[i%len(mseq)],markersize=ms,markeredgewidth=mew,markeredgecolor='k',markevery=mark[i])
    ax.tick_params(labelsize='large',width=3); 
    ax.set_xlabel('Iteration',fontsize=22)
    ax.set_ylabel("Distance to equilibrium",fontsize=22)
    ax.margins(x=0,y=0); 
    ax.ticklabel_format(useOffset=False,useMathText=True,scilimits=(-5,1),style="scientific")
    ax.tick_params(labelsize=16); ax.grid(alpha=0.3); 
    ax.legend(fontsize=14,frameon=True, bbox_to_anchor=(0, -.3, 1, -.5), loc="lower left", mode="expand", borderaxespad=0, ncol=5,labelspacing=0.12)
    ax.xaxis.get_offset_text().set_fontsize(16); 
    ax.set_title(title,fontsize=20,pad=10); 
    #ax.set_yscale('log', base=10)
    plt.tight_layout()
    for ext,dpi in [("png",800),("pdf",None),("svg",None)]:
        p=f"{outfile_base}.{ext}"; 
        (fig.savefig(p,bbox_inches='tight',dpi=dpi) if dpi else fig.savefig(p,bbox_inches='tight'))
    plt.close(fig)

def plot_distances_wc_time (curves, title, outfile_base):
    set_style(); alpha=0.9; lw=3.0; mew=0.7; ms=9
    palette=[colors20[8], colors20[0], colors20[2], colors20[14],colors20[4], colors20[6], colors20[10], colors20[12]]
    mseq=[markers[3], markers[6], markers[-1], markers[0],  markers[4], markers[2], markers[5], markers[1]]
    mark=[100, 150, 200, 250, 300, 350]
    fig,ax=plt.subplots(figsize=(13,6)); 
    for i,(name,y,wc_times) in enumerate(curves):
        it=np.arange(len(y))
        wc_times=wc_times
        ax.plot(wc_times,y,lw=lw,color=palette[i%len(palette)],label=name,alpha=alpha)
    ax.tick_params(labelsize='large',width=3); 
    ax.set_xlabel('Wall-clock time (s)',fontsize=22)
    ax.set_ylabel("Distance to equilibrium",fontsize=22)
    ax.margins(x=0,y=0); 
    ax.ticklabel_format(useOffset=False,useMathText=True,scilimits=(-5,1),style="scientific")
    ax.tick_params(labelsize=16); ax.grid(alpha=0.3); 
    ax.legend(fontsize=14,frameon=True, bbox_to_anchor=(0, -.3, 1, -.5), loc="lower left", mode="expand", borderaxespad=0, ncol=5,labelspacing=0.12)
    ax.xaxis.get_offset_text().set_fontsize(16); 
    ax.set_title(title,fontsize=20,pad=10); 
    ax.set_yscale('log', base=10)
    #plt.tight_layout()
    for ext,dpi in [("png",800),("pdf",None),("svg",None)]:
        p=f"{outfile_base}_wc_time.{ext}"; 
        (fig.savefig(p,bbox_inches='tight',dpi=dpi) if dpi else fig.savefig(p,bbox_inches='tight'))
    plt.close(fig)