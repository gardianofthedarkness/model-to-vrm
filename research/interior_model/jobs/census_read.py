import sys, json, csv, io
sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
from vault import get
from huggingface_hub import HfApi, hf_hub_download
tok = get('huggingface'); api = HfApi(token=tok); repo = '%s/totalsegmentator-ct-v201' % api.whoami()['name']
print('repo', repo, 'private', api.repo_info(repo, repo_type='dataset').private)
for f in api.list_repo_tree(repo, repo_type='dataset', recursive=True):
    print(' ', f.path, getattr(f, 'size', ''))
s = json.load(open(hf_hub_download(repo, 'census/summary.json', repo_type='dataset', token=tok, local_dir='census_dl')))
print('labels', len(s['labels']), s['labels']); print('other', s['other_files'])
m = open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset', token=tok, local_dir='census_dl'), encoding='utf-8-sig').read().splitlines()
print('meta rows', len(m) - 1, 'header:', m[0]); 
rows = list(csv.DictReader(open(hf_hub_download(repo, 'census/heads.csv', repo_type='dataset', token=tok, local_dir='census_dl'))))
B = lambda r, k: r[k] == 'True'
def cnt(f): return sum(1 for r in rows if f(r))
print('heads', len(rows))
print('bottom ok + C3', cnt(lambda r: not B(r,'skull_cut_bottom') and B(r,'c3_present')))
print('  + front not cropped', cnt(lambda r: not B(r,'skull_cut_bottom') and B(r,'c3_present') and not B(r,'face_cropped_front')))
print('  + side not cropped', cnt(lambda r: not B(r,'skull_cut_bottom') and B(r,'c3_present') and not B(r,'face_cropped_front') and not B(r,'face_cropped_side')))
print('  + top not cut', cnt(lambda r: not B(r,'skull_cut_bottom') and B(r,'c3_present') and not B(r,'face_cropped_front') and not B(r,'skull_cut_top')))
import numpy as np
p = np.array([float(r['max_soft_tissue_in_front_of_skull_mm']) for r in rows if not B(r,'skull_cut_bottom') and B(r,'c3_present') and not B(r,'face_cropped_front')])
print('protrusion in that group pct', np.percentile(p,[5,25,50,75,95]).round(1) if len(p) else None)
