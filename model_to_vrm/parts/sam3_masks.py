"""SAM 3 raw instance masks per close-up and concept, saved once so every voting rule can be tried offline:
  python sam3_masks.py <views dir> [render,render,...]   (default: every render in cameras.json)   ->  <dir>/masks/<render>.npz
Per concept: masks (n, R*R/8 packed bits) and scores (n). Every instance SAM 3 returns is kept (threshold 0); which ones
count is decided later from the pooled scores. Renders already saved are skipped."""
import sys, os, json, time, numpy as np, torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
D = sys.argv[1]; cams = json.load(open(os.path.join(D, 'cameras.json'))); ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else sorted(cams)
ONLY = [n for n in ONLY if (cams[n].get('band') or n.split('_')[0]) != 'clay' or len(sys.argv) > 2]   # clay body views are for pose, not parts
od = os.path.join(D, 'masks'); os.makedirs(od, exist_ok=True)
CONCEPTS = {'head': ['face', 'hair', 'eye', 'eyebrow', 'mouth', 'nose', 'ear'],
            'eyes': ['eye', 'iris', 'pupil', 'eye white', 'eyelash', 'eye highlight', 'eyelid'],
            'torso': ['button', 'badge', 'cuff stripe', 'pocket stripe', 'pocket'],
            'face': ['eye', 'iris', 'eyelid', 'eyebrow', 'mouth', 'lips', 'nose', 'ear'],
            # whole body, textured: what is skin and what is cloth or a carried thing; any model, unused ones drop out later
            'tex': ['skin', 'face', 'hair', 'hand', 'arm', 'leg', 'neck', 'shirt', 'jacket', 'skirt', 'pants', 'dress', 'sock',
                    'shoe', 'bag', 'tie', 'belt', 'collar', 'hat', 'glasses', 'bow']}   # bare-geometry close-ups (geo_views.py)
proc = Sam3Processor.from_pretrained('facebook/sam3'); model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.float16).eval().to('cuda'); t0 = time.time()
for k, name in enumerate(ONLY):
    f = os.path.join(od, name + '.npz')
    if os.path.exists(f): continue
    cm = cams[name]; cm.setdefault('band', name.split('_')[0]); im = Image.open(os.path.join(D, name + '.png')).convert('RGB')
    x = proc(images=im, return_tensors='pt').to('cuda'); sizes = x['original_sizes'].tolist(); res = {}
    with torch.no_grad():
        ve = model.get_vision_features(pixel_values=x['pixel_values'].half())
        for c in CONCEPTS[cm['band']]:
            t = proc(text=c, return_tensors='pt').to('cuda'); o = model(vision_embeds=ve, input_ids=t['input_ids'], attention_mask=t['attention_mask'])
            r = proc.post_process_instance_segmentation(o, threshold=0.0, mask_threshold=0.5, target_sizes=sizes)[0]
            M = r['masks'].cpu().numpy().astype(bool); S = r['scores'].float().cpu().numpy()
            res['m_' + c] = np.packbits(M.reshape(len(M), -1), axis=1); res['s_' + c] = S
    np.savez_compressed(f, **res)
    print('SM %2d/%d %s: %s  %.0f s' % (k + 1, len(ONLY), name, ' '.join('%s %d>%.2f' % (c, len(res['s_' + c]), res['s_' + c].max() if len(res['s_' + c]) else 0) for c in CONCEPTS[cm['band']]), time.time() - t0), flush=True)
print('SM done')
