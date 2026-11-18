"""Download public expression/reference data; preserve existing verified inputs."""
from pathlib import Path
from urllib.request import urlopen
import shutil,json,hashlib
ROOT=Path(__file__).parent;DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
PUBLIC='https://gdc-hub.s3.us-east-1.amazonaws.com/download/'
NAMES=['TCGA-LUAD.star_tpm.tsv.gz','TCGA-LUAD.star_tpm.tsv.json','TCGA-LUAD.survival.tsv.json','TCGA-LUAD.clinical.tsv.gz','TCGA-LUAD.survival.tsv.gz','gencode.v36.annotation.gtf.gene.probemap']
SETS=['HALLMARK_GLYCOLYSIS','HALLMARK_OXIDATIVE_PHOSPHORYLATION','HALLMARK_FATTY_ACID_METABOLISM','HALLMARK_CHOLESTEROL_HOMEOSTASIS','HALLMARK_BILE_ACID_METABOLISM','HALLMARK_PEROXISOME','HALLMARK_ADIPOGENESIS','HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY','HALLMARK_MTORC1_SIGNALING']
urls={n:PUBLIC+n for n in NAMES};urls.update({s+'.json':'https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/'+s+'.json' for s in SETS})
def main():
 for name,url in urls.items():
  p=DATA/name
  if not p.exists():
   print('Downloading',name,flush=True);tmp=p.with_name(p.name+'.partial')
   with urlopen(url,timeout=180) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
   tmp.replace(p)
 audit=[]
 for p in DATA.iterdir():
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
   audit.append({'file':p.name,'url':urls.get(p.name,'User-supplied Hallmark Hypoxia MSigDB v2026.2 or reference file'),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 (ROOT/'provenance.json').write_text(json.dumps(audit,indent=2));print('Inputs ready. Run python pipeline.py')
if __name__=='__main__':main()
