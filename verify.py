"""Scientific integrity checks: score mathematics, sample independence, splits, model roundtrip."""
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from pipeline import score_matrix,bh

def main():
 root=Path(__file__).parent;out=root/'results';checks={}
 # Independent hand calculation of weighted ssGSEA ES for a no-tie toy profile.
 n=30;names=[f'GENE{i:02d}' for i in range(n)];v=np.arange(1,n+1,dtype=float);hits=set(names[18:]);a=pd.DataFrame({'toy':v},index=names)
 observed=score_matrix(a,{'TOY_HYPOXIA':list(hits)}).iloc[0,0];order=np.argsort(-v);weights=v**.25;den=sum(weights[i] for i,g in enumerate(names) if g in hits);balance=0.;area=0.
 for i in order:
  balance+=weights[i]/den if names[i] in hits else -1/(n-len(hits));area+=balance
 checks['ssgsea_matches_independent_weighted_area_formula']=bool(np.isclose(observed,area,atol=1e-6))
 b=a.assign(other=v[::-1]);paired=score_matrix(b,{'TOY_HYPOXIA':list(hits)}).loc['toy'].iloc[0];checks['raw_ES_sample_independent']=bool(np.isclose(observed,paired))
 checks['BH_known_example']=bool(np.allclose(bh([.01,.04,.03,.002]),[.02,.04,.04,.008]))
 s=pd.read_csv(out/'prediction_split.csv');dev=set(s[s.role=='development'].patient);test=set(s[s.role=='untouched_test'].patient);checks['patient_split_disjoint']=not bool(dev&test)
 bundle=joblib.load(out/'frozen_model.joblib');checks['selected_genes_exclude_score_genes']=not bool(set(bundle['selected_genes'])&set(bundle['hypoxia_genes']))
 before=joblib.hash(bundle['pipeline']);shape=(4,len(bundle['input_genes']));bundle['pipeline'].predict_proba(np.zeros(shape));after=joblib.hash(bundle['pipeline']);checks['inference_does_not_fit_or_mutate_pipeline']=before==after
 pp=pd.read_csv(out/'patient_scores_and_clinical.csv');checks['one_score_per_patient']=pp.patient.is_unique and pp.hypoxia_score.notna().all() and len(pp)==516
 pred=pd.read_csv(out/'untouched_test_predictions.csv');checks['all_test_predictions_from_reserved_patients']=set(pred.patient)<=test
 checks['reported_results_match_predictions']=abs(__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(pred.true_derived_label,pred.probability_high)-json.loads((out/'summary.json').read_text())['modelling']['holdout']['roc_auc'])<1e-12
 (out/'verification_checks.json').write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2));print(json.dumps(checks,indent=2,default=bool));assert all(checks.values())
if __name__=='__main__':main()
