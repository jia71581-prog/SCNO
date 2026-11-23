from pathlib import Path
import importlib.util,sys,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SHARED=ROOT/'shared'
for p in (SHARED,):sys.path.insert(0,str(p))
def upstream(name):
 p=SHARED/(name+'.py');s=importlib.util.spec_from_file_location('r13_'+name,p);v=importlib.util.module_from_spec(s);sys.modules[s.name]=v;s.loader.exec_module(v);return v
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
