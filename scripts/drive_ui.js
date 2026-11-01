// Drives the running app over the Chrome DevTools protocol (start it with --remote-debugging-port=9222):
// loads models, starts a text-only call, injects fragments, prints the tick tape and ends the call.
//   npx electron . --remote-debugging-port=9222   (in another terminal)
//   node scripts/drive_ui.js
const SCRIPT = ["Hey I'm cleaning", 'the bathroom', 'right now and', 'I got some', 'bleach here', '', "and I'm going", 'to mix it',
  'with the ammonia', 'cleaner so it', 'gets extra strong', '', '', '', '', '', ''];

(async () => {
  const pages = await (await fetch('http://127.0.0.1:9222/json')).json();
  const page = pages.find((p) => p.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0; const pending = new Map();
  ws.onmessage = (m) => { const j = JSON.parse(m.data); if (j.id && pending.has(j.id)) { pending.get(j.id)(j); pending.delete(j.id); } };
  const send = (method, params) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expr) => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails)); return r.result.result.value; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const status = () => evaluate(`[...document.querySelectorAll('.pill')].map(p => p.textContent).join(' | ')`);

  console.log('title:', await evaluate('document.title'), '| status:', await status());
  await evaluate(`document.querySelector('#mic-on').checked = false`);
  await evaluate(`document.querySelector('#btn-services').click()`);
  for (let i = 0; i < 120; i++) { await sleep(1000); const s = await status(); if (/llama: ready/.test(s) && /tts: ready/.test(s)) break; if (i % 5 === 0) console.log('  ', s); }
  console.log('after load:', await status());
  await evaluate(`document.querySelector('#btn-call').click()`);
  await sleep(1500);
  console.log('after start:', await status());
  for (const frag of SCRIPT) {
    if (frag) await evaluate(`window.talkme.inject(${JSON.stringify(frag)})`);
    await sleep(600);
  }
  await sleep(1500);
  const tape = await evaluate(`[...document.querySelectorAll('#tape tbody tr')].map(r => [...r.children].map(c => c.textContent).join(' | ')).join('\\n')`);
  console.log('tick tape:\n' + tape);
  console.log('stats:', await evaluate(`document.querySelector('#tick-stats').textContent`));
  console.log('transcript:', await evaluate(`document.querySelector('#transcript').innerText.replace(/\\n/g, ' / ')`));
  await evaluate(`document.querySelector('#btn-call').click()`);
  await sleep(500);
  console.log('after stop:', await status());
  const log = await evaluate(`document.querySelector('#log').textContent.split('\\n').filter(l => /error|Error|failed|uncaught/.test(l)).slice(-10).join('\\n')`);
  console.log('errors in app log:', log || '(none)');
  ws.close();
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
