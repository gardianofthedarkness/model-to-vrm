"""Run the Tripo CLI (npm tripo-cli) with the API key from the system's credential store (vault.py name 'trio'), never from a file.
usage: python tripo_run.py <tripo arguments...>        e.g. python tripo_run.py balance
The key is handed to the CLI only in this process's environment (TRIPO_API_KEY, the CLI's highest-priority auth), so the CLI never
writes it to its own config (~/.tripo). Node's install folder and the global npm folder are put on PATH for the child."""
import os, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vault import get
key = get('trio')
if not key: raise SystemExit('no Tripo key in the vault (vault.py name: trio)')
env = dict(os.environ, TRIPO_API_KEY=key)
env['PATH'] = os.pathsep.join([r'C:\Program Files\nodejs', os.path.join(os.environ.get('APPDATA', ''), 'npm'), env.get('PATH', '')])
exe = shutil.which('tripo', path=env['PATH']) or shutil.which('tripo.cmd', path=env['PATH'])
if not exe: raise SystemExit('tripo CLI not found: npm install -g tripo-cli')
raise SystemExit(subprocess.call([exe] + sys.argv[1:], env=env))
