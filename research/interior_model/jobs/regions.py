import sys, csv, collections, numpy as np
sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
from vault import get
from huggingface_hub import hf_hub_download, HfApi
tok = get('huggingface'); repo = '%s/totalsegmentator-ct-v201' % HfApi(token=tok).whoami()['name']
p = hf_hub_download(repo, 'census/file_list.tsv', repo_type='dataset', token=tok, local_dir='census_dl')
sz = collections.defaultdict(dict)
for line in open(p, encoding='utf-8'):
    n, f = line.rstrip('\n').split('\t', 1)
    if '/segmentations/' in f and f.endswith('.nii.gz'):
        c = f.split('/')[-3]; L = f.rsplit('/', 1)[1][:-7]; sz[c][L] = int(n) - len(L)   # gzip header stores the file name
# empty-mask size per case: the smallest mask in that case; present = clearly larger
def empty_size(d):
    cnt = collections.Counter(d.values()); v, k = cnt.most_common(1)[0]
    return v if k >= 2 else -1          # all-zero masks of one case compress to identical bytes
pres = {c: {k for k, v in d.items() if v != empty_size(d)} for c, d in sz.items()}
print('cases with no shared empty size (all labels present or unknown):', sum(empty_size(d) < 0 for d in sz.values()))
# sanity: how often is the smallest mask tied (several empties of equal size)?
ties = np.mean([sorted(d.values())[1] <= 1.02 * min(d.values()) for d in sz.values()])
print('cases', len(sz), 'share of cases with >=2 near-equal smallest masks (empty-size estimate reliable):', round(ties, 2))
meta = {r['image_id']: r for r in csv.DictReader(open('census_dl/census/meta.csv', encoding='utf-8-sig'), delimiter=';')}
def n(f): return sum(1 for c, s in pres.items() if f(s, meta.get(c, {})))
V = lambda pre, a, b: {'vertebrae_%s%d' % (pre, i) for i in range(a, b + 1)}
R = {
 'both femurs present': lambda s, m: {'femur_left', 'femur_right'} <= s,
 'femur + study type includes leg': lambda s, m: 'femur_left' in s and 'leg' in m.get('study_type', ''),
 'whole pelvis (both hips, sacrum, L5)': lambda s, m: {'hip_left', 'hip_right', 'sacrum', 'vertebrae_L5'} <= s,
 'whole trunk spine (C7..L5 all present)': lambda s, m: V('T', 1, 12) | V('L', 1, 5) | {'vertebrae_C7'} <= s,
 'thorax (all 24 ribs + sternum)': lambda s, m: {'rib_%s_%d' % (a, i) for a in ('left', 'right') for i in range(1, 13)} | {'sternum'} <= s,
 'one shoulder (scapula, clavicle, humerus same side)': lambda s, m: any({'scapula_' + x, 'clavicula_' + x, 'humerus_' + x} <= s for x in ('left', 'right')),
 'both shoulders': lambda s, m: all({'scapula_' + x, 'clavicula_' + x, 'humerus_' + x} <= s for x in ('left', 'right')),
 'neck (C1..C7 all present)': lambda s, m: V('C', 1, 7) <= s,
 'skull present': lambda s, m: 'skull' in s,
}
for k, f in R.items(): print('%-55s %d' % (k, n(f)))
