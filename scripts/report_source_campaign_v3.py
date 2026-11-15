"""Export per-head/horizon evaluation plots for the new independent source campaign."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from astronaut_twin.source_expansion import OUT

if __name__=='__main__':
    metrics=pd.read_csv(OUT/'metrics.csv');specs=json.loads((OUT/'specs.json').read_text(encoding='utf8'))
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    for source,spec in specs.items():
        data=metrics[(metrics.source==source)&(metrics.split=='test')]
        for variable,label in zip(spec['targets'],spec['labels']):
            selected=data[data.variable==variable]
            fig,axes=plt.subplots(1,2,figsize=(10,3.5))
            for method,color in [('persistence','#b896ae'),('transformer','#7aabc7'),('ensemble','#68a890')]:
                frame=selected[selected.method==method].sort_values('horizon')
                axes[0].plot(frame.horizon,frame.mae,'o-',label=method,color=color)
                axes[1].plot(frame.horizon,frame.coverage,'o-',label=method,color=color)
            axes[0].set_ylabel('MAE · original source units');axes[1].set_ylabel('Observed interval coverage');axes[1].axhline(.8,color='#b37c50',linestyle='--',label='nominal 80%');axes[1].set_ylim(0,1.05)
            for ax in axes:ax.set_xlabel('Horizon · '+('hours' if spec['clock']=='hours' else 'visits'));ax.set_xticks(spec['horizons']);ax.grid(alpha=.15)
            axes[1].legend(fontsize=7);fig.suptitle(source+' · '+label+' · exploratory held-out evaluation',fontsize=10);fig.tight_layout()
            safe=''.join(c if c.isalnum() else '_' for c in variable);fig.savefig(directory/(source+'_'+safe+'.png'),dpi=150);plt.close(fig)
    print('Plots:',len(list(directory.glob('*.png'))))
