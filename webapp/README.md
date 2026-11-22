# Model to VRM — web app

Run the pipeline from a browser (phone friendly), check every step's pictures, redo any step. No Claude needed.

**Start:** double-click or run
`powershell -ExecutionPolicy Bypass -File E:\code\model-to-vrm\webapp\start_web.ps1`
It starts the server (127.0.0.1:8765, its own minimised window) and a tunnel, and prints **PHONE LINK** — open it on the
phone once; the key is then remembered in a cookie for 30 days.

**Use:** pick a model (or *Add a model* → upload a `.glb`), then *Run next step* to go one step at a time and check its
pictures, or *Run all remaining*. *Redo* reruns one step; *From here* reruns it and everything after it. One job runs at a
time; *Stop* ends it. Logs are under each step.

**Where things are:** uploads and results in `E:\figures\runs\<name>\`; Rikka and test model R are listed from
`E:\figures\runs\runs.json`. The access key is `E:\figures\runs\.webkey` — delete it and restart to make a new one
(old links stop working).

**Limits, honestly:**
- The tunnel is a Cloudflare quick tunnel (`E:\tools\cloudflared\cloudflared.exe`, signed by Cloudflare): free, no
  account, but a new address every time the script runs, and it stops when this PC sleeps or the window closes. Run the
  script again for a new link. (A fixed address needs a free Cloudflare account and a named tunnel.)
- A full run takes about 1–2 hours on this laptop; SAM 3 (GPU) and the LLM steps are the slow ones.
- The LLM steps call DeepSeek with the key from the vault; they stop if a run would make more than `LLM_MAX_CALLS`
  (200) calls.
- Not in the web pipeline yet: the CT anatomy slices (still research, `notebooks/anatomy_slices.ipynb`).
