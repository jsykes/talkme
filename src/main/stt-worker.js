'use strict';
// Runs the streaming recogniser off the main thread. In: { type: 'audio', samples } (16 kHz float32) and
// { type: 'reset' }. Out: { type: 'result', text, endpoint } after every decoded chunk, { type: 'ready' }, { type: 'error' }.
const { parentPort, workerData } = require('worker_threads');
const path = require('path');
const fs = require('fs');

let rec = null, stream = null, punct = null, lastText = '';
try {
  const sherpa = require('sherpa-onnx-node');
  const cfg = workerData;
  const dir = cfg.dir;
  const pick = (prefix) => {
    const all = fs.readdirSync(dir).filter((f) => f.startsWith(prefix) && f.endsWith('.onnx'));
    const f = all.find((x) => !x.includes('int8')) || all[0];
    if (!f) throw new Error(`no ${prefix}*.onnx in ${dir}`);
    return path.join(dir, f);
  };
  // NeMo FastConformer models (Nemotron) use 128 mel bins; icefall zipformers use 80
  const featureDim = cfg.featureDim || (/nemotron|parakeet|nemo/i.test(dir) ? 128 : 80);
  rec = new sherpa.OnlineRecognizer({
    featConfig: { sampleRate: 16000, featureDim },
    modelConfig: {
      transducer: { encoder: pick('encoder'), decoder: pick('decoder'), joiner: pick('joiner') },
      tokens: path.join(dir, 'tokens.txt'), numThreads: cfg.threads || 4, provider: 'cpu', debug: 0,
    },
    decodingMethod: 'greedy_search',
    // endpointing only resets the recogniser's utterance; the LLM never sees it as a turn boundary
    enableEndpoint: true, rule1MinTrailingSilence: 2.4, rule2MinTrailingSilence: 1.2, rule3MinUtteranceLength: 20,
  });
  stream = rec.createStream();
  if (cfg.punctDir && fs.existsSync(path.join(cfg.punctDir, 'model.onnx'))) {
    punct = new sherpa.OnlinePunctuation({ model: { cnnBilstm: path.join(cfg.punctDir, 'model.onnx'), bpeVocab: path.join(cfg.punctDir, 'bpe.vocab'), numThreads: 1, provider: 'cpu' } });
  }
  parentPort.postMessage({ type: 'ready', featureDim, punct: !!punct });
} catch (e) {
  parentPort.postMessage({ type: 'error', error: e.message });
}

parentPort.on('message', (m) => {
  if (!rec) return;
  if (m.type === 'close') { rec = null; stream = null; punct = null; parentPort.close(); return; }
  try {
    if (m.type === 'reset') { rec.reset(stream); lastText = ''; return; }
    if (m.type === 'punct') { // punctuate a text on request (only used for upper-case, unpunctuated recognisers)
      parentPort.postMessage({ type: 'punct', id: m.id, text: punct ? punct.addPunct(m.text) : m.text });
      return;
    }
    if (m.type !== 'audio') return;
    stream.acceptWaveform({ sampleRate: 16000, samples: m.samples });
    while (rec.isReady(stream)) rec.decode(stream);
    const text = rec.getResult(stream).text.trim();
    const endpoint = rec.isEndpoint(stream);
    if (text !== lastText || endpoint) {
      let punctuated = null;
      // recognisers that emit UPPER CASE with no punctuation get the punctuation model applied here
      if (punct && text && text === text.toUpperCase() && /[A-Z]/.test(text)) punctuated = punct.addPunct(text.toLowerCase());
      parentPort.postMessage({ type: 'result', text, punctuated, endpoint });
      lastText = text;
    }
    if (endpoint) { rec.reset(stream); lastText = ''; }
  } catch (e) {
    parentPort.postMessage({ type: 'error', error: e.message });
  }
});
