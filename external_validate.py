"""Apply frozen model to an independent, gene-symbol-by-sample log2(TPM+1) matrix.
No independent cohort was run in the delivered project. All frozen score-background
symbols are required to avoid comparing scores calculated on different universes.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from pipeline import score_matrix
from model import labels,metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--expression',required=True,type=Path);ap.add_argument('--model',type=Path,default=Path(__file__).parent/'results/frozen_model.joblib');ap.add_argument('--out',type=Path,default=Path('external_results'));ap.add_argument('--unit',required=True,choices=['log2(TPM+1)']);ap.add_argument('--independent-patients',action='store_true',required=True);a=ap.parse_args()
 b=joblib.load(a.model);e=pd.read_csv(a.expression,sep='\t',index_col=0);e=e.groupby(level=0).mean()
 if not np.isfinite(e.to_numpy()).all() or (e.to_numpy()<0).any():raise ValueError('Invalid expression values')
 if set(e.columns)&(set(b['training_sample_ids'])|set(b['test_sample_ids'])):raise ValueError('Known source samples detected; independent patients required')
 missing=set(b['score_background'])-set(e.index)
 if missing:raise ValueError(f'{len(missing)} frozen background symbols missing. Do not silently impute or change the score universe; harmonize data and predefine a compatible validation protocol.')
 e=e.reindex(b['score_background']);score=score_matrix(e,{'HALLMARK_HYPOXIA':b['hypoxia_genes']})['HALLMARK_HYPOXIA'];y=labels(score.to_numpy(),b['cutoffs']);prob=b['pipeline'].predict_proba(e.loc[b['input_genes']].T.to_numpy())[:,1]
 r=pd.DataFrame({'sample':e.columns,'hypoxia_ES':score.values,'derived_label':y,'probability_high':prob,'prediction':(prob>=.5).astype(int)});a.out.mkdir(exist_ok=True,parents=True);r.to_csv(a.out/'external_predictions.csv',index=False)
 keep=y>=0
 if len(np.unique(y[keep]))==2:(a.out/'metrics.json').write_text(json.dumps(metrics(y[keep],prob[keep]),indent=2))
 else:(a.out/'metrics.json').write_text(json.dumps({'status':'Insufficient two-class extreme-group labels; predictions exported only.'}))
 print(a.out)
if __name__=='__main__':main()
