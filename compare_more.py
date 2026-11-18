"""Expanded exploratory benchmark. Original holdout has already been inspected."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,SGDClassifier
import model
from pipeline import load_data
ROOT=Path(__file__).parent

def extra_specs():
 return {
 'ExtraTrees':(ExtraTreesClassifier(n_estimators=200,class_weight='balanced',min_samples_leaf=3,n_jobs=2,random_state=2026),{'model__max_depth':[5,None]}),
 'HistGradientBoosting':(HistGradientBoostingClassifier(max_iter=150,l2_regularization=5,random_state=2026),{'model__max_leaf_nodes':[7,15]}),
 'AdaBoost':(AdaBoostClassifier(n_estimators=150,random_state=2026),{'model__learning_rate':[.1,.5]}),
 'GradientBoosting':(GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=2026),{'model__learning_rate':[.03,.1]}),
 'GaussianNB':(GaussianNB(),{'model__var_smoothing':[1e-9,1e-5,.01]}),
 'ShrinkageLDA':(LinearDiscriminantAnalysis(solver='lsqr'),{'model__shrinkage':['auto',.5]}),
 'ElasticNetLogistic':(LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,class_weight='balanced',max_iter=8000,random_state=2026),{'model__C':[.1,1]}),
 'SGDLogistic':(SGDClassifier(loss='log_loss',penalty='elasticnet',class_weight='balanced',max_iter=5000,tol=1e-5,random_state=2026),{'model__alpha':[.001,.01]})}

def main():
 out=ROOT/'results/additional_benchmark';out.mkdir(exist_ok=True)
 expr,_=load_data(ROOT/'data',out);patients=pd.read_csv(ROOT/'results/patient_scores_and_clinical.csv',index_col=0);expr=expr.reindex(columns=patients.index)
 hypo=(ROOT/'data/HALLMARK_HYPOXIA.v2026.2.Hs.gmt').read_text().strip().split('\t')[2:];sets={}
 for f in (ROOT/'data').glob('HALLMARK_*.json'):
  for k,v in json.loads(f.read_text()).items():sets[k]=v['geneSymbols']
 model.model_specs=extra_specs
 summary=model.run_models(expr,patients,sets,hypo,out)
 summary['status']='Expanded exploratory benchmark: original holdout was already inspected. Selection remains based on development CV; do not treat repeated holdout use as fresh confirmation.'
 (out/'expanded_summary.json').write_text(json.dumps(summary,indent=2))
 original=pd.read_csv(ROOT/'results/nested_cv_fold_metrics.csv');new=pd.read_csv(out/'nested_cv_fold_metrics.csv');allfold=pd.concat([original,new],ignore_index=True);allfold.to_csv(out/'all_14_classifier_fold_metrics.csv',index=False)
 tab=allfold.groupby('model').agg(mean_AUC=('roc_auc','mean'),sd_AUC=('roc_auc','std'),balanced_accuracy=('balanced_accuracy','mean'),macro_F1=('macro_f1','mean'),accuracy=('accuracy','mean')).sort_values('mean_AUC',ascending=False).reset_index();tab.to_csv(out/'all_14_classifier_comparison.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,6));t=tab.sort_values('mean_AUC');ax.barh(t.model,t.mean_AUC,xerr=t.sd_AUC,color='#44839c');ax.set(xlim=(0,1.05),xlabel='Development outer-fold ROC AUC',title='14 classifiers: same development splits; error bars = fold SD');fig.tight_layout();fig.savefig(out/'all_14_comparison.png',dpi=150);plt.close(fig)
 import base64
 pic=base64.b64encode((out/'all_14_comparison.png').read_bytes()).decode();clusters=pd.read_csv(ROOT/'results/cluster_metrics.csv')
 html=f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Expanded LUAD Model Comparison</title><style>body{{font:16px/1.6 system-ui;max-width:1050px;margin:35px auto;padding:20px;color:#243442}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ddd;padding:7px;text-align:left}}th{{background:#eaf1f5}}img{{max-width:100%}}.note{{background:#fff1d6;padding:16px}}</style><h1>Comparing 14 classifiers and clustering methods</h1><p>All classifiers use metabolic gene-expression inputs outside the hypoxia-scoring set. The same five outer development folds and three inner tuning folds are used. Feature filtering, scaling and selection are fitted within training folds.</p><p class="note">This is an expanded exploratory comparison. The original test set has already been inspected. The table below uses development cross-validation, not test-set ranking. Independent cohort validation remains pending. The original frozen model is preserved; expanded candidate models are stored separately.</p><img src="data:image/png;base64,{pic}" alt="14 classifier development comparison"><h2>Supervised classifiers</h2>{tab.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}<p>The highest development mean belongs to <strong>{tab.iloc[0]['model']}</strong>. Close scores and overlapping fold variation do not establish a statistically meaningful winner. These models predict RNA-derived high/low labels, not measured oxygen.</p><h2>Clustering is a separate question</h2><p>K-Means, hierarchical clustering and DBSCAN group similar tumors without receiving hypoxia labels. Their cluster IDs are compared afterward with score-derived groups. Silhouette measures geometric separation; ARI and AMI measure label agreement. These are not classifier accuracy or ROC AUC and should not be ranked directly against the classifier table.</p>{clusters.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}<h2>How to reproduce</h2><p>Run the original pipeline first, then <code>python compare_more.py</code>. Full fold-level results, expanded candidate model and parameters are in <code>results/additional_benchmark</code>. The original pipeline and frozen model remain in <code>results</code>.</p></html>'''
 (out/'comparison_report.html').write_text(html)
 print(tab.to_string(index=False),flush=True)
if __name__=='__main__':main()
