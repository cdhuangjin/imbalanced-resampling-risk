"""External guard for the declared R1 process tree; kills only that verified tree."""
import json
from pathlib import Path
import time
import psutil

root_dir = Path(__file__).resolve().parents[1]
matches = []
for proc in psutil.process_iter(['cmdline', 'name']):
    cmd = proc.info['cmdline'] or []
    if proc.info['name'].lower().startswith('python') and any(s.replace('\\', '/').endswith('15_elor_smote/code/run_reproduction.py') for s in cmd):
        matches.append(proc)
if len(matches) != 1:
    raise RuntimeError(f'Expected one R1 parent process; found {len(matches)}')
parent = matches[0]
started = parent.create_time()
cpu_by_pid = {}
out = root_dir / 'outputs' / 'r1'
log = out / 'budget_monitor.jsonl'
while parent.is_running():
    try:
        processes = [parent] + parent.children(recursive=True)
        rss = 0
        for p in processes:
            try:
                c = p.cpu_times()
                cpu_by_pid[p.pid] = max(cpu_by_pid.get(p.pid, 0), c.user+c.system)
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        state = dict(time=time.time(), pid=parent.pid, elapsed=time.time()-started, cpu_seconds=sum(cpu_by_pid.values()), rss=rss, output_bytes=size)
        reasons = [name for name, crossed in [('wall',state['elapsed']>4*3600),('cpu',state['cpu_seconds']>8*3600),('ram',rss>8*1024**3),('disk',size>2*1024**3)] if crossed]
        state['violations'] = reasons
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(state)+'\n')
        if reasons:
            for p in reversed(processes):
                try:
                    p.kill()
                except psutil.Error:
                    pass
            print(json.dumps({'status':'BUDGET_STOP','violations':reasons}),flush=True)
            break
        time.sleep(5)
    except psutil.NoSuchProcess:
        break
else:
    print(json.dumps({'status':'PROCESS_ENDED_WITHIN_OBSERVED_BUDGET'}),flush=True)
