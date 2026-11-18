"""Reproducible TCGA-LUAD hypoxia research pipeline. See README before interpreting results."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse,json,hashlib,platform,warnings,time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score,adjusted_rand_score,adjusted_mutual_info_score
from model import run_models
from survival import run_survival
SEED=2026
plt.rcParams.update({'figure.dpi':130,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
COLORS={'Low':'#2878a5','Intermediate':'#a4a9ad','High':'#d55546'}
def log(t):print(time.strftime('%H:%M:%S'),t,flush=True)
def bh(p):
 p=np.asarray(p,float);out=np.full(len(p),np.nan);ok=np.where(np.isfinite(p))[0];order=ok[np.argsort(p[ok])];m=len(order)
 if m:out[order]=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1].clip(0,1)
 return out
def dump(obj,path):
 Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def groups(scores,cutoffs=None):
 lo,hi=np.quantile(scores,[1/3,2/3]) if cutoffs is None else cutoffs
 return pd.Series(np.where(scores<=lo,'Low',np.where(scores>=hi,'High','Intermediate')),index=scores.index),[float(lo),float(hi)]
def score_matrix(expr,sets):
 # ES is sample-specific. NES rescales across samples and is deliberately not used.
 s=gp.ssgsea(data=expr,gene_sets=sets,sample_norm_method='rank',correl_norm_type='rank',weight=.25,min_size=10,max_size=1000,permutation_num=0,outdir=None,no_plot=True,threads=2,seed=SEED)
 z=s.res2d.pivot(index='Name',columns='Term',values='ES').astype(float)
 return z.reindex(expr.columns)
def load_data(data,out):
 expr=pd.read_csv(data/'TCGA-LUAD.star_tpm.tsv.gz',sep='\t',index_col=0,dtype={})
 mapping=pd.read_csv(data/'gencode.v36.annotation.gtf.gene.probemap',sep='\t').set_index('id')
 clinical=pd.read_csv(data/'TCGA-LUAD.clinical.tsv.gz',sep='\t');surv=pd.read_csv(data/'TCGA-LUAD.survival.tsv.gz',sep='\t')
 assert expr.shape==(60660,589),f'Unexpected source shape: {expr.shape}'
 assert expr.columns.is_unique and expr.index.is_unique
 values=expr.to_numpy();assert np.isfinite(values).all() and (values>=0).all(),'Invalid expression values'
 assert json.loads((data/'TCGA-LUAD.star_tpm.tsv.json').read_text())['unit']=='log2(tpm+1)'
 meta=pd.DataFrame({'sample':expr.columns});meta['patient']=meta['sample'].str[:12];meta['code']=meta['sample'].str[13:15]
 primary=meta[meta.code=='01'].copy();primary['selected']=~primary.sort_values('sample').duplicated('patient')
 # Fixed lexical sample selection. Does not use expression, stage or survival.
 selected=sorted(primary[primary.selected]['sample']);primary.to_csv(out/'sample_selection.csv',index=False)
 assert len(selected)==516 and len(set(x[:12] for x in selected))==516
 raw=expr[selected].copy();symbols=mapping.reindex(raw.index)['gene'];assert symbols.notna().all()
 raw.index=symbols.astype(str)
 # Average duplicate-symbol log TPM rows: deterministic rule applied identically to any external cohort.
 expr=raw.groupby(level=0,sort=True).mean().astype(np.float32)
 clinical=clinical.drop_duplicates('sample').set_index('sample');surv=surv.drop_duplicates('sample').set_index('sample')
 patients=pd.DataFrame(index=selected);patients.index.name='sample';patients['patient']=[s[:12] for s in selected]
 assert patients.index.isin(clinical.index).all()
 cs=clinical.reindex(selected)
 patients['age']=pd.to_numeric(cs['age_at_index.demographic'],errors='coerce')
 patients['sex']=cs['gender.demographic'];patients['stage_original']=cs['ajcc_pathologic_stage.diagnoses']
 def stage(s):
  if pd.isna(s):return np.nan
  for roman,n in [('IV',4),('III',3),('II',2),('I',1)]:
   if str(s).startswith('Stage '+roman):return n
  return np.nan
 patients['stage']=patients.stage_original.map(stage)
 patients['pack_years']=pd.to_numeric(cs['pack_years_smoked.exposures'],errors='coerce')
 patients['OS.time']=pd.to_numeric(surv.reindex(selected)['OS.time'],errors='coerce');patients['OS']=pd.to_numeric(surv.reindex(selected)['OS'],errors='coerce')
 zz=surv.join(clinical,on='sample',rsuffix='_clinical');cross=pd.crosstab(zz.OS,zz['vital_status.demographic'])
 assert cross.loc[0,'Dead']==0 and cross.loc[1,'Alive']==0
 timechecks={}
 for event,col in [(0,'days_to_last_follow_up.diagnoses'),(1,'days_to_death.demographic')]:
  w=zz[zz.OS==event];v=pd.to_numeric(w[col],errors='coerce');ok=v.notna()&w['OS.time'].notna();timechecks[str(event)]={'checked':int(ok.sum()),'equal_days':int(np.isclose(v[ok],w.loc[ok,'OS.time']).sum())}
 dump({'source_genes':60660,'source_samples':589,'primary_samples':528,'selected_unique_patients':len(selected),'gene_symbols':len(expr),'duplicate_symbols_collapsed':len(raw)-len(expr),'expression_unit':'log2(TPM+1)','survival_coding':'0 censored/alive; 1 death; OS.time in days, checked against clinical day fields','survival_crosscheck':timechecks,'sample_selection':'lexicographically first primary tumor sample per patient','symbol_collapse':'mean log2(TPM+1) across duplicate gene symbols'},out/'data_audit.json')
 return expr,patients

def explorations(expr,patients,hypoxia,out):
 # All-cohort exploratory plots are distinct from all training-fitted predictive transforms.
 X=expr.T;eligible=(expr>=1).mean(axis=1)>=.1
 X=X.loc[:,eligible & ~expr.index.isin(hypoxia)]
 X=X.loc[:,X.var().nlargest(min(5000,X.shape[1])).index]
 scaler=StandardScaler();z=scaler.fit_transform(X)
 pca=PCA(n_components=50,random_state=SEED,svd_solver='randomized');pcs=pca.fit_transform(z)
 np.savetxt(out/'pca_explained_variance.csv',np.c_[np.arange(1,51),pca.explained_variance_ratio_,pca.explained_variance_ratio_.cumsum()],delimiter=',',header='component,variance_fraction,cumulative',comments='')
 coords=pd.DataFrame(pcs,index=X.index,columns=[f'PC{i+1}' for i in range(50)]);coords.to_csv(out/'pca_coordinates.csv')
 def scatter(a,title,file):
  fig,ax=plt.subplots(figsize=(6,4.5))
  for g in ['Low','Intermediate','High']:
   mask=patients.group.eq(g).values;ax.scatter(a[mask,0],a[mask,1],s=16,alpha=.65,label=g,color=COLORS[g])
  ax.set(title=title,xlabel='Component 1',ylabel='Component 2');ax.legend(title='Estimated hypoxia');fig.savefig(out/file);plt.close(fig)
 scatter(pcs,'PCA of expression outside the scoring gene set','pca.png')
 fig,ax=plt.subplots(figsize=(6,3));ax.plot(np.arange(1,51),pca.explained_variance_ratio_.cumsum());ax.set(xlabel='Number of components',ylabel='Cumulative variance explained',ylim=(0,1));fig.savefig(out/'pca_scree.png');plt.close(fig)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');ica=FastICA(n_components=10,whiten='unit-variance',random_state=SEED,max_iter=2000,tol=.005).fit_transform(pcs[:,:30]);scatter(ica,'Independent components of expression','ica.png')
  dump([str(w.message) for w in ws],out/'ica_warnings.json')
 ts=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=SEED,max_iter=1000).fit_transform(pcs[:,:30]);scatter(ts,'t-SNE of expression','tsne.png')
 import umap
 um=umap.UMAP(n_neighbors=20,min_dist=.2,random_state=SEED,n_jobs=1).fit_transform(pcs[:,:30]);scatter(um,'UMAP of expression','umap.png')
 pd.DataFrame({'UMAP1':um[:,0],'UMAP2':um[:,1],'tSNE1':ts[:,0],'tSNE2':ts[:,1]},index=X.index).to_csv(out/'nonlinear_coordinates.csv')
 # Fixed 20 PCs retains more information than plotting two dimensions. Report retained fraction.
 clX=StandardScaler().fit_transform(pcs[:,:20]);rows=[];labels={}
 for k in range(2,9):
  m=KMeans(n_clusters=k,n_init=30,random_state=SEED).fit(clX);lab=m.labels_;labels[f'KMeans_{k}']=lab
  rows.append({'method':f'KMeans_{k}','clusters':k,'silhouette':silhouette_score(clX,lab),'inertia':m.inertia_,'ARI':adjusted_rand_score(patients.group,lab),'AMI':adjusted_mutual_info_score(patients.group,lab),'noise':0})
 best=max(rows,key=lambda r:r['silhouette'])['method']
 for method,lab in [('Hierarchical_3',AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(clX))]+[(f'DBSCAN_{ep}',DBSCAN(eps=ep,min_samples=8).fit_predict(clX)) for ep in [3,4,5,6]]:
  labels[method]=lab;mask=lab>=0;nc=len(set(lab[mask]));sil=silhouette_score(clX[mask],lab[mask]) if nc>1 and mask.sum()>nc else np.nan
  rows.append({'method':method,'clusters':nc,'silhouette':sil,'inertia':np.nan,'ARI':adjusted_rand_score(patients.group,lab),'AMI':adjusted_mutual_info_score(patients.group,lab),'noise':int((~mask).sum())})
 pd.DataFrame(rows).to_csv(out/'cluster_metrics.csv',index=False);pd.DataFrame(labels,index=X.index).to_csv(out/'cluster_assignments.csv')
 ct=pd.crosstab(pd.Series(labels[best],index=X.index,name='cluster'),patients.group);ct.to_csv(out/'cluster_hypoxia_counts.csv')
 fig,ax=plt.subplots(figsize=(5,3));sns.heatmap(ct,annot=True,fmt='d',cmap='Blues',ax=ax);ax.set_title(f'{best}: original hypoxia groups');fig.savefig(out/'cluster_counts.png');plt.close(fig)
 # Distribution checks are descriptive, not a gate for choosing a test post hoc.
 normal=[]
 for gene in X.columns[:12]:
  v=X[gene].values;w,p=stats.shapiro(v);normal.append({'gene':gene,'W':w,'p_value':p})
 pd.DataFrame(normal).to_csv(out/'normality_examples.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(10,3))
 for ax,gene in zip(axes,X.columns[:3]):stats.probplot(X[gene],dist='norm',plot=ax);ax.set_title(gene)
 fig.tight_layout();fig.savefig(out/'qq_examples.png');plt.close(fig)
 stability=[]
 base=labels[best];k=int(best.split('_')[-1])
 for seed in range(10):
  lab=KMeans(n_clusters=k,n_init=10,random_state=seed).fit_predict(clX);stability.append(adjusted_rand_score(base,lab))
 dump({'best_kmeans_by_silhouette':best,'clustering_PCs':20,'variance_in_20PCs':float(pca.explained_variance_ratio_[:20].sum()),'seed_stability_ARI':stability,'hypoxia_scoring_genes_excluded':True},out/'exploration_summary.json')
 return {'best_cluster':best,'pca_first_two_variance':float(pca.explained_variance_ratio_[:2].sum())}

def biology(expr,patients,sets,hypoxia,out):
 mask=(expr>=1).mean(axis=1)>=.1;xx=expr.loc[mask];high=xx.loc[:,patients.group.eq('High')];low=xx.loc[:,patients.group.eq('Low')]
 tt=stats.ttest_ind(high,low,axis=1,equal_var=False,nan_policy='raise');mw=stats.mannwhitneyu(high,low,axis=1,alternative='two-sided',method='asymptotic')
 de=pd.DataFrame({'gene':xx.index,'mean_high_logTPM':high.mean(axis=1).values,'mean_low_logTPM':low.mean(axis=1).values,'difference_log2_TPMplus1':high.mean(axis=1).values-low.mean(axis=1).values,'welch_t':tt.statistic,'welch_p':tt.pvalue,'mannwhitney_p':mw.pvalue})
 de['welch_fdr']=bh(de.welch_p);de['mannwhitney_fdr']=bh(de.mannwhitney_p);de['scoring_gene']=de.gene.isin(hypoxia)
 metabolic=set().union(*sets.values());de['metabolic_gene']=de.gene.isin(metabolic)
 de['direction']=np.where(de.difference_log2_TPMplus1>0,'Higher','Lower');de.sort_values('welch_fdr').to_csv(out/'differential_expression.csv.gz',index=False)
 sig=de[(de.welch_fdr<.05)&(de.difference_log2_TPMplus1.abs()>=.5)&~de.scoring_gene]
 sig.to_csv(out/'differing_genes_outside_scoring_set.csv',index=False)
 universe=set(de.loc[~de.scoring_gene,'gene']);ora=[]
 for direction in ['Higher','Lower']:
  hits=set(sig.loc[sig.direction==direction,'gene'])
  for name,gs in sets.items():
   gs=set(gs)&universe;over=hits&gs;M=len(universe);n=len(gs);N=len(hits);k=len(over)
   p=stats.hypergeom.sf(k-1,M,n,N) if N and n else 1
   ora.append({'pathway':name,'direction':direction,'tested_background':M,'pathway_genes':n,'selected_genes':N,'overlap':k,'p_value':p,'overlapping_genes':';'.join(sorted(over))})
 ora=pd.DataFrame(ora);ora['fdr']=bh(ora.p_value);ora.to_csv(out/'metabolic_pathway_enrichment.csv',index=False)
 fig,ax=plt.subplots(figsize=(6,4));ax.scatter(de.difference_log2_TPMplus1,-np.log10(de.welch_fdr.clip(lower=1e-300)),s=3,c=np.where(de.scoring_gene,'#cccccc',np.where(de.metabolic_gene,'#c24c40','#6889a0')),alpha=.5);ax.axhline(-np.log10(.05),ls='--',color='grey');ax.set(xlabel='Difference in mean log2(TPM + 1): high minus low',ylabel='−log10 adjusted p',title='Descriptive expression comparison');fig.savefig(out/'volcano.png');plt.close(fig)
 top=sig[sig.metabolic_gene].sort_values('welch_fdr').head(20).gene.tolist()
 if top:
  order=patients.sort_values('hypoxia_score').index;heat=xx.loc[top,order];heat=heat.sub(heat.mean(axis=1),axis=0).div(heat.std(axis=1).replace(0,1),axis=0)
  fig,ax=plt.subplots(figsize=(10,5));sns.heatmap(heat,cmap='vlag',center=0,xticklabels=False,ax=ax,vmin=-2,vmax=2);ax.set(xlabel='Tumors ordered by hypoxia score',ylabel='Gene',title='Top differing metabolic genes outside scoring set');fig.savefig(out/'gene_heatmap.png');plt.close(fig)
 # Metabolic scores omit all hypoxia-signature genes, reducing direct gene overlap.
 independent={name:sorted(set(gs)-set(hypoxia)) for name,gs in sets.items()};pathscore=score_matrix(expr,independent);pathscore.to_csv(out/'metabolic_pathway_scores.csv')
 assoc=[]
 for name in pathscore:
  rho,p=stats.spearmanr(patients.hypoxia_score,pathscore[name]);assoc.append({'pathway':name,'spearman_rho':rho,'p_value':p,'matched_genes_without_hypoxia':len(set(independent[name])&set(expr.index))})
 assoc=pd.DataFrame(assoc);assoc['fdr']=bh(assoc.p_value);assoc.to_csv(out/'pathway_hypoxia_associations.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4));z=assoc.sort_values('spearman_rho');ax.barh(z.pathway.str.replace('HALLMARK_',''),z.spearman_rho,color=np.where(z.spearman_rho>0,'#d55546','#2878a5'));ax.set(xlabel='Spearman correlation with hypoxia score',title='Metabolic gene-set scores excluding hypoxia genes');fig.tight_layout();fig.savefig(out/'pathway_correlations.png');plt.close(fig)
 return {'genes_tested':len(de),'differing_genes_outside_signature':len(sig),'differing_metabolic_genes':int(sig.metabolic_gene.sum()),'pathways_significant_fdr':int((assoc.fdr<.05).sum())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path(__file__).parent/'data');ap.add_argument('--out',type=Path,default=Path(__file__).parent/'results');ap.add_argument('--skip-exploration',action='store_true');args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 log('Loading and checking source data');expr,patients=load_data(args.data,out)
 hypoxia=(args.data/'HALLMARK_HYPOXIA.v2026.2.Hs.gmt').read_text().strip().split('\t')[2:]
 sets={}
 for f in sorted(args.data.glob('HALLMARK_*.json')):
  for k,v in json.loads(f.read_text()).items():sets[k]=v['geneSymbols']
 assert len(sets)==9
 dump({'hypoxia_total':len(hypoxia),'hypoxia_matched':len(set(hypoxia)&set(expr.index)),'hypoxia_unmatched':sorted(set(hypoxia)-set(expr.index)),'alias_policy':'no unverified alias substitution; unmatched genes omitted and reported','metabolic_sets':{k:len(v) for k,v in sets.items()}},out/'gene_set_coverage.json')
 log('Calculating sample-specific ssGSEA ES scores');scores=score_matrix(expr,{'HALLMARK_HYPOXIA':hypoxia})['HALLMARK_HYPOXIA'];patients['hypoxia_score']=scores;patients['group'],cuts=groups(scores);patients.to_csv(out/'patient_scores_and_clinical.csv')
 # Verify score of a patient is independent of other patients, using ES rather than NES.
 single=score_matrix(expr.iloc[:,:1],{'HALLMARK_HYPOXIA':hypoxia}).iloc[0,0];assert np.isclose(single,scores.iloc[0]),'Scoring depends on cohort composition'
 assert patients.hypoxia_score.notna().all()
 summary={'cohort':len(patients),'groups':patients.group.value_counts().to_dict(),'descriptive_tertile_cutoffs':cuts,'scoring':'GSEApy ssGSEA raw ES; ranks; alpha 0.25; no cohort-dependent NES','external_validation':'NOT RUN: independent cohort not supplied','checks':{'unique_patient_count':patients.patient.nunique(),'single_sample_score_invariance':True}}
 fig,ax=plt.subplots(figsize=(6,3));ax.hist(scores,bins=25,color='#6889a0');[ax.axvline(c,color='#b8453f',ls='--') for c in cuts];ax.set(xlabel='Raw ssGSEA hypoxia ES',ylabel='Patients',title='Distribution of estimated hypoxia');fig.savefig(out/'hypoxia_distribution.png');plt.close(fig)
 if not args.skip_exploration:
  log('Running PCA, ICA, t-SNE, UMAP and clustering');summary['exploration']=explorations(expr,patients,hypoxia,out)
 log('Comparing genes and metabolic pathways');summary['biology']=biology(expr,patients,sets,hypoxia,out)
 log('Training models with nested development evaluation and an untouched holdout');summary['modelling']=run_models(expr,patients,sets,hypoxia,out)
 log('Analyzing survival');summary['survival']=run_survival(patients,out)
 import importlib.metadata
 summary['versions']={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','gseapy','lifelines','xgboost','umap-learn']};summary['python']=platform.python_version();summary['seed']=SEED
 dump(summary,out/'summary.json')
 from report import build_report
 build_report(out)
 log('Complete: '+str(out/'report.html'))
if __name__=='__main__':main()
