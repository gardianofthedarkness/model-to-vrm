"""Private gallery of run pictures: every PNG in <viewer>/runs, newest first, with its caption from runs/captions.json.
Served only through serve_private.py (CT-derived pictures never leave private storage). args: <viewer dir>"""
import os, sys, json, html, time
v = sys.argv[1]; d = os.path.join(v, 'runs'); cap = json.load(open(os.path.join(d, 'captions.json'))) if os.path.exists(os.path.join(d, 'captions.json')) else {}
pics = sorted((f for f in os.listdir(d) if f.endswith('.png')), key=lambda f: -os.path.getmtime(os.path.join(d, f)))
cards = ''.join('<figure><a href="runs/%s"><img src="runs/%s" loading="lazy"></a><figcaption><b>%s</b> · %s<br>%s</figcaption></figure>' % (f, f, html.escape(f[:-4]), time.strftime('%d %b %H:%M', time.localtime(os.path.getmtime(os.path.join(d, f)))), html.escape(cap.get(f, ''))) for f in pics)
open(os.path.join(v, 'runs.html'), 'w', encoding='utf8').write('''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Interior model runs</title><style>body{margin:0;background:#14161a;color:#e8e6e1;font:14px system-ui,sans-serif;padding:16px}h1{font-size:18px}
figure{margin:0 0 22px;background:#1c1f25;border:1px solid #2c313a;border-radius:10px;padding:10px}img{max-width:100%%;border-radius:6px;background:#fff}figcaption{color:#aab;margin-top:6px;font-size:13px}</style></head>
<body><h1>Interior model: run pictures (private)</h1><p style="color:#aab">Newest first. Tap a picture for full size. Reload for new runs.</p>%s</body></html>''' % cards)
print(len(pics), 'pictures')
