'use strict';
// Child processes that start their own (uv starts python, the pipeline starts llama-server) are stopped with
// everything they started: taskkill /T on Windows; on Linux each runs in its own process group, signalled as a whole.
const { spawn, execFileSync } = require('child_process');

const GROUP = process.platform !== 'win32';

function spawnTree(cmd, args, opts = {}) {
  return spawn(cmd, args, { windowsHide: true, detached: GROUP, ...opts });
}

function killTree(p) {
  if (!p || p.exitCode !== null) return;
  if (!GROUP) { try { execFileSync('taskkill', ['/PID', String(p.pid), '/T', '/F'], { windowsHide: true }); } catch {} return; }
  try { process.kill(-p.pid, 'SIGTERM'); } catch { try { p.kill(); } catch {} }
}

// Stops what listens on 127.0.0.1:`port` if its program is `image` (Windows: netstat and tasklist; Linux: ss), and
// nothing else: for a server that outlived the tree it was started in. Returns the pid it stopped, or null.
function killPortOwner(port, image) {
  try {
    let pid = null, name = '';
    if (process.platform === 'win32') {
      const out = execFileSync('netstat', ['-ano', '-p', 'TCP'], { encoding: 'utf8', windowsHide: true });
      const row = out.split(/\r?\n/).find((l) => /LISTENING/.test(l) && new RegExp(`(127\\.0\\.0\\.1|0\\.0\\.0\\.0):${port}\\s`).test(l));
      if (!row) return null;
      pid = Number(row.trim().split(/\s+/).pop());
      name = execFileSync('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true }).split('","')[0].replace(/^"/, '');
    } else {
      const m = /\(\("([^"]+)",pid=(\d+)/.exec(execFileSync('ss', ['-ltnpH', `sport = :${port}`], { encoding: 'utf8' }));
      if (!m) return null;
      [name, pid] = [m[1], Number(m[2])];
    }
    if (!pid || !name.toLowerCase().startsWith(image.toLowerCase())) return null;
    if (process.platform === 'win32') execFileSync('taskkill', ['/F', '/PID', String(pid)], { windowsHide: true });
    else process.kill(pid, 'SIGTERM');
    return pid;
  } catch { return null; }
}

module.exports = { spawnTree, killTree, killPortOwner };
