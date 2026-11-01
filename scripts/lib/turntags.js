// Shared by the real-conversation converters (yap_to_calls.js, oto_to_calls.js): turn one speaker's word timings
// (the reference) and the other's (the caller) into per-tick `said` and tags for tune/evaluate.py and training.
//   start          the reference took the turn after the caller had spoken (the model should start here)
//   continuer      a short back-channel while the caller holds the floor
//   overlap_start  the reference talked over a caller who kept going (not scored)
//   self_resume    the reference carried on their own turn with no word from the caller in between (not scored)
//   barge_stop     the caller cut in and the reference gave way: going quiet here is right
//   listen_speech / listen_pause / listen_last / listen_idle   the reference is quiet
const TICK = 0.6, LISTEN = '<listen>';
// words that, alone or two or three together, make a back-channel rather than a turn
const BC = /^(yeah|yea|yes|yep|yup|mhm|mm-?hmm?|mm|uh-?huh|um-?hum|right|okay|ok|oh|ah|wow|really|sure|true|exactly|nice|cool|totally|no|nope|hmm|huh|god|damn|gosh|haha|ha|aw|aww|definitely|absolutely|correct|same|facts|interesting)$/i;

// userWords / refWords: [{ word, start, end }] in seconds on the conversation timeline
function buildCall(userWords, refWords, t0, dur) {
  const n = Math.floor(dur / TICK);
  const uw = userWords.filter((w) => w.end > t0 && w.start < t0 + dur);
  const rw = refWords.filter((w) => w.start >= t0 && w.start < t0 + dur);
  const tickOf = (t) => Math.floor((t - t0) / TICK);
  // who is voiced in each tick, from the word timings (the recogniser lags; the speech itself doesn't)
  const userOn = Array(n).fill(false);
  for (const w of uw) for (let k = Math.max(0, tickOf(w.start)); k <= Math.min(n - 1, tickOf(w.end)); k++) userOn[k] = true;
  const said = Array(n).fill('');
  for (const w of rw) { const k = tickOf(w.start); if (k < n) said[k] = (said[k] ? said[k] + ' ' : '') + w.word; }
  const tags = Array(n).fill(null);
  const runEnd = (i) => { let j = i; while (j + 1 < n && (said[j + 1] || (j + 2 < n && said[j + 2] && !userOn[j + 1]))) j++; return j; };
  let i = 0, prevEnd = -1;
  while (i < n) {
    if (!said[i]) { i++; continue; }
    const j = runEnd(i);
    if (!userOn.slice(prevEnd + 1, i + 1).some(Boolean)) {
      tags[i] = 'self_resume'; for (let k = i + 1; k <= j; k++) tags[k] = 'speak';
      prevEnd = j; i = j + 1; continue;
    }
    const ws = said.slice(i, j + 1).filter(Boolean).join(' ').split(/\s+/).map((w) => w.replace(/[^a-z'-]/gi, ''));
    const userBefore = userOn.slice(Math.max(0, i - 3), i + 1).some(Boolean);
    const userAfter = userOn.slice(j + 1, Math.min(n, j + 5)).some(Boolean);
    if (ws.length <= 3 && ws.every((w) => BC.test(w)) && userBefore && userAfter) {
      tags[i] = 'continuer';
      for (let k = i + 1; k <= j; k++) tags[k] = 'speak';
    } else {
      const overUser = i > 0 && userOn[i - 1] && userOn[i] && userOn.slice(i + 1, Math.min(n, i + 3)).filter(Boolean).length >= 2;
      tags[i] = overUser ? 'overlap_start' : 'start';
      for (let k = i + 1; k <= j; k++) tags[k] = 'speak';
      let cut = -1;
      for (let k = i + 2; k <= j; k++) if (userOn[k] && !userOn[k - 1]) { cut = k; break; }
      if (cut > 0 && j - cut <= 2 && j - i >= 3 && j + 1 < n) tags[j + 1] = 'barge_stop';
    }
    prevEnd = j; i = j + 1;
  }
  for (let k = 0; k < n; k++) {
    if (tags[k]) continue;
    if (userOn[k]) { tags[k] = 'listen_speech'; continue; }
    let resume = false;
    for (let m = k + 1; m < Math.min(n, k + 5); m++) { if (said[m]) break; if (userOn[m]) { resume = true; break; } }
    const recentUser = userOn.slice(Math.max(0, k - 3), k).some(Boolean);
    tags[k] = resume && recentUser ? 'listen_pause' : recentUser ? 'listen_last' : 'listen_idle';
  }
  return { tags, said: said.map((s) => s || LISTEN) };
}

module.exports = { buildCall, TICK, LISTEN, BC };
