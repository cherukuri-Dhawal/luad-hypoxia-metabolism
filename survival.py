import json,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter,CoxPHFitter
from lifelines.statistics import multivariate_logrank_test,proportional_hazard_test

def run_survival(patients,out):
 p=patients.copy();valid=p['OS'].isin([0,1])&p['OS.time'].notna()&(p['OS.time']>0);s=p.loc[valid].copy();s.to_csv(out/'survival_eligible_patients.csv')
 lr=multivariate_logrank_test(s['OS.time'],s.group,s.OS)
 fig,ax=plt.subplots(figsize=(7,4));colors={'Low':'#2878a5','Intermediate':'#a4a9ad','High':'#d55546'}
 for g in ['Low','Intermediate','High']:
  a=s[s.group==g];KaplanMeierFitter(label=f'{g} (n={len(a)})').fit(a['OS.time'],a.OS).plot_survival_function(ax=ax,ci_show=False,color=colors[g])
 ax.set(xlabel='Follow-up days',ylabel='Overall survival probability',title=f'Calculated hypoxia groups: log-rank p={lr.p_value:.3g}');fig.savefig(out/'survival_km.png');plt.close(fig)
 # Fixed full-cohort SD scaling is descriptive, not predictive. Stage is categorical, never ordinal-linear in Cox.
 s['hypoxia_per_SD']=(s.hypoxia_score-p.hypoxia_score.mean())/p.hypoxia_score.std();s['male']=s.sex.map({'male':1.,'female':0.})
 c=s[['OS.time','OS','hypoxia_per_SD','age','male','stage']].dropna().copy();c['stage']=c.stage.astype(int).astype(str);c=pd.get_dummies(c,columns=['stage'],drop_first=True,dtype=float)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');cox=CoxPHFitter(penalizer=.01);cox.fit(c,duration_col='OS.time',event_col='OS');ph=proportional_hazard_test(cox,c,time_transform='rank')
 cox.summary.to_csv(out/'cox_age_sex_stage_adjusted.csv');ph.summary.to_csv(out/'cox_proportional_hazards_check.csv')
 # Prespecified sensitivity for nuisance-covariate PH violations: separate baseline hazards by stage and sex.
 strat=s[['OS.time','OS','hypoxia_per_SD','age','male','stage']].dropna().copy()
 strat['stage']=strat.stage.astype(int).astype(str);strat['male']=strat.male.astype(int).astype(str)
 stratcox=CoxPHFitter(penalizer=.01).fit(strat,duration_col='OS.time',event_col='OS',strata=['stage','male'])
 stratcox.summary.to_csv(out/'cox_stage_sex_stratified_sensitivity.csv')
 stratph=proportional_hazard_test(stratcox,strat,time_transform='rank');stratph.summary.to_csv(out/'cox_stratified_PH_check.csv')
 stratrow=stratcox.summary.loc['hypoxia_per_SD']
 fig,ax=plt.subplots(figsize=(7,3.5));cox.plot(ax=ax);ax.set_title('Exploratory Cox model: age, sex and categorical stage');fig.savefig(out/'cox_effects.png');plt.close(fig)
 missing=s[['age','sex','stage']].isna().sum().to_dict();row=cox.summary.loc['hypoxia_per_SD']
 # Group-stage association is descriptive; expected cells may be small and are reported.
 tab=pd.crosstab(s.group,s.stage);chi=__import__('scipy').stats.chi2_contingency(tab)
 tab.to_csv(out/'stage_by_hypoxia_counts.csv')
 result={'survival_n':len(s),'death_events':int(s.OS.sum()),'excluded_no_valid_positive_followup':int((~valid).sum()),'logrank_p':float(lr.p_value),'cox_complete_case_n':len(c),'cox_death_events':int(c.OS.sum()),'cox_hypoxia_HR_per_SD':float(row['exp(coef)']),'cox_hypoxia_HR_CI95':[float(row['exp(coef) lower 95%']),float(row['exp(coef) upper 95%'])],'cox_hypoxia_p':float(row['p']),'cox_adjustment':'age, sex, stage categorical; penalizer 0.01','PH_hypoxia_p':float(ph.summary.loc['hypoxia_per_SD','p']),'missing_covariates_in_survival':missing,'stage_chisquare_p':float(chi.pvalue),'stage_min_expected_cell':float(chi.expected_freq.min()),'nuisance_PH_violations_at_05':ph.summary.index[(ph.summary.p<.05)&(ph.summary.index!='hypoxia_per_SD')].tolist(),'stratified_sensitivity':{'HR_per_SD':float(stratrow['exp(coef)']),'CI95':[float(stratrow['exp(coef) lower 95%']),float(stratrow['exp(coef) upper 95%'])],'p':float(stratrow['p']),'hypoxia_PH_p':float(stratph.summary.loc['hypoxia_per_SD','p']),'strata':'stage and sex'},'warnings':sorted(set(str(w.message) for w in ws)),'interpretation':'Exploratory association. PH violations and missingness limit interpretation; no causal or individual clinical prognosis claim.'}
 (out/'survival_summary.json').write_text(json.dumps(result,indent=2));return result
