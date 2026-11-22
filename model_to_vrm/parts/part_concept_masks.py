"""SAM 3 instance masks for one concept on the close-ups of one part (run on this machine; about 90 s a view on a 4 GB card):
  python part_concept_masks.py <parts_zoom dir> <region> <view,view,...> <concept,concept,...>
-> <parts_zoom dir>/<region>/masks/<view>.npz (per concept: packed masks m_<concept>, scores s_<concept>; every instance kept)
and an overlay sheet <region>_<concept>_sam.jpg. Views already saved are skipped."""
import sys, os, json, time, numpy as np, torch, cv2
from PIL import Image
from transformers import Sam3Processor, Sam3Model
D, REG = sys.argv[1], sys.argv[2]; views = sys.argv[3].split(','); concepts = sys.argv[4].split(','); od = os.path.join(D, REG, 'masks'); os.makedirs(od, exist_ok=True); t0 = time.time()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; dt = torch.float16 if dev == 'cuda' else torch.float32
proc = Sam3Processor.from_pretrained('facebook/sam3'); model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=dt).eval().to(dev); print('TS model on', dev, '%.0f s' % (time.time() - t0), flush=True)
tiles = []
for name in views:
    if os.path.exists(os.path.join(od, name + '.npz')): print('TS', name, 'kept'); continue
    im = Image.open(os.path.join(D, REG, name + '.png')).convert('RGB'); x = proc(images=im, return_tensors='pt').to(dev); sizes = x['original_sizes'].tolist(); res = {}; bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        ve = model.get_vision_features(pixel_values=x['pixel_values'].to(dt))
        for c in concepts:
            t = proc(text=c, return_tensors='pt').to(dev); o = model(vision_embeds=ve, input_ids=t['input_ids'], attention_mask=t['attention_mask'])
            r = proc.post_process_instance_segmentation(o, threshold=0.0, mask_threshold=0.5, target_sizes=sizes)[0]; M = r['masks'].cpu().numpy().astype(bool); S = r['scores'].float().cpu().numpy()
            res['m_' + c] = np.packbits(M.reshape(len(M), -1), axis=1); res['s_' + c] = S; o_ = np.argsort(-S); print('TS %s %-8s %d instances, scores %s (%.0f s)' % (name, c, len(S), np.round(S[o_][:8], 2).tolist(), time.time() - t0), flush=True)
            ov = bgr.copy(); rng = np.random.default_rng(0)
            for k in o_[:8]:
                if S[k] < 0.3: continue
                col = rng.integers(60, 255, 3); ov[M[k]] = (0.45 * ov[M[k]] + 0.55 * col).astype(np.uint8); ys, xs = np.nonzero(M[k])
                if len(xs): cv2.putText(ov, '%.2f' % S[k], (int(xs.mean()) - 20, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(ov, '%s: "%s"' % (name, c), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA); tiles.append(cv2.resize(ov, (512, 512)))
    np.savez_compressed(os.path.join(od, name + '.npz'), res=im.size[0], **res)
n = len(concepts)
if not tiles: sys.exit(0)
rows = [np.hstack(tiles[i:i + n]) for i in range(0, len(tiles), n)]; cv2.imwrite(os.path.join(D, REG + '_%s_sam.jpg' % concepts[0]), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print('TS done %.0f s' % (time.time() - t0))
