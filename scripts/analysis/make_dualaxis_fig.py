import pandas as pd, numpy as np, glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
RESULTS=BASE/'results'
DIRS={'DeepSeek-V3.2':['DeepSeek','deepseek-v3.2-cloud-batch'],
      'Gemma-4-31B':['Gemma','gemma4-31b-cloud-batch'],
      'Llama-4-Maverick':['Nvidia','meta-llama-4-maverick-17b-batch'],
      'Mixtral-8x22B':['Mixtral','mixtral_8x22b'],
      'Nemotron-Ultra-253B':['Nvidia','llama-3.1-nemotron-ultra-253b-v1-batch-off']}
# tiny -> ultra order with a distinct colour each
ORDER=['Gemma-4-31B','Mixtral-8x22B','Nemotron-Ultra-253B','Llama-4-Maverick','DeepSeek-V3.2']
COLORS={'Gemma-4-31B':'#2ca02c','Mixtral-8x22B':'#ff7f0e','Nemotron-Ultra-253B':'#17becf',
        'Llama-4-Maverick':'#d62728','DeepSeek-V3.2':'#1f77b4'}
TASKS=[('Quaternary_Easy','Quaternary'),('Quinary','Quinary')]
BATCHES=[1,2,4,8,16,32,64]
# Two aggregations shown as the two rows of the figure.
MODES=[('ezs','EZS prompt'),('avg','Average of both prompts')]

def load(model,task,mode):
    p=RESULTS/DIRS[model][1]/task/'batch_summary_aggregated.csv'
    d=pd.read_csv(p)
    if mode=='ezs':
        d=d[d['Prompt']=='Enhanced_Zero_Shot']
    # for 'avg' we keep both prompts and average them per batch size below
    d=d.groupby('Batch Size',as_index=False).mean(numeric_only=True)
    return d.set_index('Batch Size').reindex(BATCHES)

# Per-panel F1 range so the inverted-U (Quinary) and general-U (Quaternary)
# shapes read clearly while the dashed cost lines use the right axis.
F1_YLIM={'Quaternary':(0.70,0.89),'Quinary':(0.58,0.77)}

fig,axes=plt.subplots(2,2,figsize=(12,8.2))
for r,(mkey,mlabel) in enumerate(MODES):
    for c,(tkey,tname) in enumerate(TASKS):
        ax=axes[r,c]; ax2=ax.twinx()
        for m in ORDER:
            d=load(m,tkey,mkey); x=np.arange(len(BATCHES))
            ax.plot(x,d['Avg Macro F1'],color=COLORS[m],lw=2,marker='o',ms=4,label=m)
            ax2.plot(x,d['Avg Tokens/Req'],color=COLORS[m],lw=1.6,ls='--',alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(BATCHES)
        ax.grid(True,alpha=0.3)
        ax.set_ylim(*F1_YLIM[tname])
        if r==0:
            ax.set_title(tname,fontsize=13,fontweight='bold')
        if r==len(MODES)-1:
            ax.set_xlabel('Batch Size',fontsize=12)
        if c==0:
            ax.set_ylabel(f'{mlabel}\n\nMacro $F_1$-score',fontsize=12)
        if c==1:
            ax2.set_ylabel('Tokens / Requirement',fontsize=12)

model_handles=[Line2D([0],[0],color=COLORS[m],lw=2.2,label=m) for m in ORDER]
style_handles=[Line2D([0],[0],color='black',lw=2,ls='-',label='Macro $F_1$ (solid, left)'),
               Line2D([0],[0],color='black',lw=1.6,ls='--',label='Tokens/Req (dashed, right)')]
fig.legend(handles=model_handles+style_handles,loc='upper center',ncol=4,
           bbox_to_anchor=(0.5,1.07),fontsize=10,frameon=False)
fig.tight_layout(rect=[0,0,1,0.97])
out=BASE/'analysis_output'/'Global_Averages'/'DualAxis_F1_Tokens_Trends.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out,dpi=300,bbox_inches='tight')
print('saved',out)
