// Reads the app's dropdowns and any page errors over DevTools (start the app with: npx electron . --remote-debugging-port=9222).
//   node scripts/check_ui.js
(async () => {
  const pages = await (await fetch('http://127.0.0.1:9222/json')).json();
  const ws = new WebSocket(pages.find((p) => p.type === 'page').webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0; const pend = new Map(); const errors = [];
  ws.onmessage = (m) => {
    const j = JSON.parse(m.data);
    if (j.method === 'Runtime.exceptionThrown') errors.push(j.params.exceptionDetails.exception?.description || j.params.exceptionDetails.text);
    if (j.id && pend.has(j.id)) { pend.get(j.id)(j); pend.delete(j.id); }
  };
  const send = (method, params) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (e) => { const j = await send('Runtime.evaluate', { expression: e, awaitPromise: true, returnByValue: true }); return j.result.result ? j.result.result.value : j; };
  await send('Runtime.enable');
  await send('Page.reload');
  await new Promise((r) => setTimeout(r, 2500));
  for (const sel of ['#mic-device', '#spk-device', '#voice', '#adapter']) {
    const opts = await ev(`[...document.querySelectorAll('${sel} option')].map((o) => o.textContent)`);
    console.log(sel.padEnd(13), `${opts.length} options:`, JSON.stringify(opts.slice(0, 4)));
  }
  console.log('duck toggle:', await ev(`document.querySelector('#duck') ? (document.querySelector('#duck').checked ? 'on' : 'off') : 'missing'`));
  console.log('page errors:', errors.length ? errors : 'none');
  ws.close();
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
