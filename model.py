"""All predictive feature learning occurs inside training folds; final test is reserved."""
import json,warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix,RocCurveDisplay,ConfusionMatrixDisplay
from xgboost import XGBClassifier
SEED=2026
class GeneFilter(BaseEstimator,TransformerMixin):
 def __init__(self,max_features=5000,min_fraction=.1,min_log_tpm=1.):self.max_features=max_features;self.min_fraction=min_fraction;self.min_log_tpm=min_log_tpm
 def fit(self,X,y=None):
  X=np.asarray(X);eligible=(X>=self.min_log_tpm).mean(axis=0)>=self.min_fraction;v=X.var(axis=0);eligible&=v>1e-12
  ix=np.where(eligible)[0];self.indices_=ix[np.argsort(v[ix],kind='stable')[-self.max_features:]]
  if not len(self.indices_):raise ValueError('No train-eligible genes')
  return self
 def transform(self,X):return np.asarray(X)[:,self.indices_]
class SafeSelect(BaseEstimator,TransformerMixin):
 def __init__(self,k=150):self.k=k
 def fit(self,X,y):self.selector_=SelectKBest(f_classif,k=min(self.k,X.shape[1])).fit(X,y);return self
 def transform(self,X):return self.selector_.transform(X)
def labels(scores,cuts):return np.where(scores<=cuts[0],0,np.where(scores>=cuts[1],1,-1))
def metrics(y,p):
 pred=(np.asarray(p)>=.5).astype(int)
 return {'n':len(y),'n_high':int(np.sum(y)),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'roc_auc':roc_auc_score(y,p),'confusion_matrix':confusion_matrix(y,pred).tolist()}
def make_pipeline(model):return Pipeline([('filter',GeneFilter()),('scale',StandardScaler()),('select',SafeSelect()),('model',model)])
def model_specs():
 return {
 'LogisticRegression':(LogisticRegression(class_weight='balanced',max_iter=4000,random_state=SEED),{'model__C':[.1,1,10]}),
 'LinearSVM':(SVC(kernel='linear',probability=True,class_weight='balanced',random_state=SEED),{'model__C':[.01,.1,1]}),
 'RBFSVM':(SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=SEED),{'model__C':[.1,1,10]}),
 'RandomForest':(RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=2),{'model__max_depth':[4,None]}),
 'XGBoost':(XGBClassifier(n_estimators=150,learning_rate=.04,subsample=.8,colsample_bytree=.8,reg_lambda=10,eval_metric='logloss',random_state=SEED,n_jobs=2),{'model__max_depth':[2,3]}),
 'kNN':(KNeighborsClassifier(weights='distance'),{'model__n_neighbors':[5,15,25]})}
def best_fit(name,X,y):
 estimator,grid=model_specs()[name];cv=StratifiedKFold(3,shuffle=True,random_state=SEED)
 search=GridSearchCV(make_pipeline(estimator),grid,cv=cv,scoring='roc_auc',n_jobs=1,refit=True,error_score='raise')
 search.fit(X,y);return search.best_estimator_,search.best_params_
def run_models(expr,patients,sets,hypoxia,out):
 metabolic=set().union(*map(set,sets.values()));genes=sorted((metabolic-set(hypoxia))&set(expr.index));X=expr.loc[genes].T.to_numpy();scores=patients.hypoxia_score.to_numpy();global_groups=patients.group.to_numpy();ids=np.arange(len(patients))
 dev,test=train_test_split(ids,test_size=.2,stratify=global_groups,random_state=SEED);assert not set(dev)&set(test)
 split=pd.DataFrame({'sample':patients.index,'patient':patients.patient.values,'role':np.where(np.isin(ids,dev),'development','untouched_test')});split.to_csv(out/'prediction_split.csv',index=False)
 outer=StratifiedKFold(5,shuffle=True,random_state=SEED);folds=list(outer.split(dev,global_groups[dev]));rows=[];predictions=[];params=[]
 for name in model_specs():
  for fold,(tr0,va0) in enumerate(folds,1):
   train,val=dev[tr0],dev[va0];cuts=np.quantile(scores[train],[1/3,2/3]);yt=labels(scores[train],cuts);yv=labels(scores[val],cuts);train=train[yt>=0];val=val[yv>=0];yt=yt[yt>=0];yv=yv[yv>=0]
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always');pipe,bp=best_fit(name,X[train],yt)
   proba=pipe.predict_proba(X[val])[:,1];r=metrics(yv,proba);r.update(model=name,fold=fold);rows.append(r)
   for ix,y,pr in zip(val,yv,proba):predictions.append({'sample':patients.index[ix],'model':name,'fold':fold,'label':int(y),'probability_high':float(pr),'train_cut_low':cuts[0],'train_cut_high':cuts[1]})
   params.append({'model':name,'fold':fold,'parameters':bp,'warnings':sorted(set(str(w.message) for w in ws))})
   print(f'  {name} fold {fold}: AUC={r["roc_auc"]:.3f}, n={len(val)}',flush=True)
 results=pd.DataFrame(rows);results.drop(columns='confusion_matrix').to_csv(out/'nested_cv_fold_metrics.csv',index=False)
 avg=results.groupby('model')[['roc_auc','balanced_accuracy','macro_f1','accuracy']].agg(['mean','std']);avg.to_csv(out/'nested_cv_model_comparison.csv')
 pd.DataFrame(predictions).to_csv(out/'development_out_of_fold_predictions.csv',index=False)
 means=results.groupby('model').roc_auc.mean().sort_values(ascending=False);best=means.index[0]
 # Selection locked by DEVELOPMENT scores before inspecting final-test outcomes.
 cuts=np.quantile(scores[dev],[1/3,2/3]);ydev=labels(scores[dev],cuts);ytest=labels(scores[test],cuts);tr=dev[ydev>=0];te=test[ytest>=0];yt=ydev[ydev>=0];yv=ytest[ytest>=0]
 pipe,bp=best_fit(best,X[tr],yt);prob=pipe.predict_proba(X[te])[:,1];held=metrics(yv,prob)
 # Baselines use the same patients and same training-derived label definition.
 dummy=DummyClassifier(strategy='prior').fit(X[tr],yt);held['dummy']=metrics(yv,dummy.predict_proba(X[te])[:,1])
 clinical=patients[['age','stage','pack_years']].copy();clinical['male']=patients.sex.map({'male':1,'female':0});clinical=clinical.to_numpy(float)
 cb=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('model',LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=SEED))]);cb.fit(clinical[tr],yt);held['clinical_only_baseline']=metrics(yv,cb.predict_proba(clinical[te])[:,1])
 # Fixed-prediction patient bootstrap reflects test-sample uncertainty, not training uncertainty.
 rng=np.random.default_rng(SEED);boot=[]
 for _ in range(2000):
  ix=rng.integers(0,len(yv),len(yv))
  if len(np.unique(yv[ix]))==2:boot.append(roc_auc_score(yv[ix],prob[ix]))
 held['auc_bootstrap_95CI']=np.quantile(boot,[.025,.975]).tolist()
 frame=pd.DataFrame({'sample':patients.index[te],'patient':patients.patient.iloc[te].values,'true_derived_label':yv,'probability_high':prob,'predicted_label':(prob>=.5).astype(int)});frame.to_csv(out/'untouched_test_predictions.csv',index=False)
 filt=pipe.named_steps['filter'].indices_;select=pipe.named_steps['select'].selector_.get_support();selected=np.array(genes)[filt][select]
 m=pipe.named_steps['model'];importance=np.abs(m.coef_[0]) if hasattr(m,'coef_') else m.feature_importances_ if hasattr(m,'feature_importances_') else np.repeat(np.nan,len(selected))
 pd.DataFrame({'gene':selected,'model_weight_or_importance':importance}).sort_values('model_weight_or_importance',ascending=False).to_csv(out/'selected_predictive_genes.csv',index=False)
 # Bundle contains no fitted preprocessing from the holdout and no arbitrary cohort-dependent NES.
 bundle={'pipeline':pipe,'input_genes':genes,'selected_genes':selected.tolist(),'hypoxia_genes':hypoxia,'score_background':expr.index.tolist(),'cutoffs':cuts.tolist(),'model_name':best,'expression_unit':'log2(TPM+1)','symbol_collapse':'mean log2(TPM+1)','training_sample_ids':patients.index[tr].tolist(),'test_sample_ids':patients.index[test].tolist(),'label_definition':'sample-specific raw ssGSEA ES with frozen development tertiles','score_weight':.25,'seed':SEED}
 joblib.dump(bundle,out/'frozen_model.joblib',compress=3)
 (out/'frozen_model_settings.json').write_text(json.dumps({k:v for k,v in bundle.items() if k not in ['pipeline','score_background','hypoxia_genes']},indent=2))
 fig,ax=plt.subplots(figsize=(5,4));RocCurveDisplay.from_predictions(yv,prob,ax=ax,name=best);ax.plot([0,1],[0,1],ls='--',c='grey');ax.set_title('Untouched patient holdout');fig.savefig(out/'holdout_roc.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(4,3.5));ConfusionMatrixDisplay.from_predictions(yv,prob>=.5,display_labels=['Low','High'],ax=ax,cmap='Blues',colorbar=False);ax.set_title('Untouched patient holdout');fig.savefig(out/'holdout_confusion.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.5));mu=results.groupby('model').roc_auc.mean();sd=results.groupby('model').roc_auc.std();ax.bar(mu.index,mu,yerr=sd,color='#6889a0');ax.set(ylim=(0,1.05),ylabel='Mean outer-fold ROC AUC',title='Development comparison: metabolic predictors only');ax.tick_params(axis='x',rotation=25);fig.savefig(out/'model_comparison.png');plt.close(fig)
 (out/'model_tuning.json').write_text(json.dumps(params,indent=2))
 # Research validity checks, beyond file existence.
 check={'development_test_disjoint':not bool(set(dev)&set(test)),'no_scoring_genes_in_inputs':not bool(set(genes)&set(hypoxia)),'no_scoring_genes_in_selected':not bool(set(selected)&set(hypoxia)),'frozen_training_test_patient_disjoint':not bool(set(patients.patient.iloc[tr])&set(patients.patient.iloc[test])),'roundtrip_predictions_equal':bool(np.allclose(joblib.load(out/'frozen_model.joblib')['pipeline'].predict_proba(X[te])[:,1],prob))}
 assert all(check.values());(out/'predictive_integrity_checks.json').write_text(json.dumps(check,indent=2))
 return {'selected_model':best,'selection_basis':'highest development nested CV mean ROC AUC; final test not used for selection','development_patients':len(dev),'test_patients_reserved':len(test),'development_extremes_used':len(tr),'test_extremes_evaluated':len(te),'test_intermediate_not_evaluated':int((ytest<0).sum()),'metabolic_input_genes_outside_hypoxia':len(genes),'selected_genes':len(selected),'best_parameters':bp,'frozen_cutoffs':cuts.tolist(),'holdout':held,'external_validation':'pending','integrity_checks':check}
