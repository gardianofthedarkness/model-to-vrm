"""Run the Pi coding agent on DeepSeek with the key from the vault passed only to Pi's process (never written to disk):
  python pi_run.py <pi arguments...>      e.g. python pi_run.py -p @card.png "What is outlined?"
Pi's ~/.pi/agent/models.json names the variable DEEPSEEK_API_KEY, not the key. Model: deepseek/deepseek-flash (sees images)."""
import os, sys, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from vault import get
env = dict(os.environ, DEEPSEEK_API_KEY=get('deepseek'), PATH=r'C:\Program Files\nodejs;' + os.environ.get('PATH', ''))
PI = shutil.which('pi.cmd', path=env['PATH']) or shutil.which('pi', path=env['PATH']) or os.path.join(os.environ.get('APPDATA', ''), 'npm', 'pi.cmd')
args = sys.argv[1:]
if '--model' not in args: args = ['--model', 'deepseek/deepseek-flash'] + args
sys.exit(subprocess.run([PI] + args, env=env, stdin=subprocess.DEVNULL).returncode)   # no stdin: -p would wait on it
