from __future__ import annotations
import argparse,json,os
from pathlib import Path
from MAS_DAG import generate_candidate_suite

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path('data/mmlu_pro/sample.jsonl')); p.add_argument('--output',type=Path,default=Path('data/mmlu_pro/candidate_graphs.json')); p.add_argument('--node-pool',type=Path,default=Path('data/node_pools/mmlu_pro_6_roles.json')); p.add_argument('--seed',type=int,default=42); p.add_argument('--random-count',type=int,default=5); a=p.parse_args()
 pool=json.load(open(a.node_pool)); nodes=pool['nodes']; fin=next(i for i,n in enumerate(nodes) if n['id']==pool['finalizer_id']); order=tuple(i for i in range(len(nodes)) if i!=fin)+(fin,); ref=os.path.relpath(a.node_pool.resolve(),a.output.resolve().parent); out=[]
 for qi,line in enumerate(open(a.input)):
  s=json.loads(line); opts='\n'.join(f"({chr(65+i)}) {x}" for i,x in enumerate(s['options'])); task=f"{s['question']}\n\nOptions:\n{opts}"; graphs=[]
  for gi,t in enumerate(generate_candidate_suite(len(nodes),fin,random_count=a.random_count,seed=a.seed+qi,fixed_order=order)):
   g=t.to_graph_record(); g['id']=f'q{qi:04d}_g{gi:02d}'; graphs.append(g)
  out.append({'task':task,'reference_answer':s['answer'],'reference_solution':s.get('cot_content',''),'source_metadata':{k:s[k] for k in ('question_id','category','src','answer_index')},'sampling_seed':a.seed+qi,'node_pool':ref,'evaluator':'mmlu_pro','graphs':graphs})
 a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_suffix('.json.tmp'); json.dump(out,open(tmp,'w'),ensure_ascii=False,indent=2); tmp.replace(a.output); print(len(out),sum(len(x['graphs']) for x in out))
if __name__=='__main__': main()
