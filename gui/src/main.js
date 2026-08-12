import './style.css'
import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'

let currentProcessId = null
let processLogElement = null

function el(id) {
  return document.getElementById(id)
}

async function refreshLists() {
  try {
    const projectDir = await invoke('get_project_dir')
    logLine(`[INFO] project_dir: ${projectDir}`)

    const gguf = await invoke('list_gguf').catch(e => { logLine(`[ERR] list_gguf: ${e}`); return [] })
    logLine(`[INFO] gguf=${gguf.length}`)

    const datasets = await invoke('list_datasets').catch(e => { logLine(`[ERR] list_datasets: ${e}`); return [] })
    logLine(`[INFO] datasets=${datasets.length}`)

    const caches = await invoke('list_caches').catch(e => { logLine(`[ERR] list_caches: ${e}`); return [] })
    logLine(`[INFO] caches=${caches.length}`)

    populateSelect('weights-select', gguf, true)
    populateDatasetList(datasets)
    populateSelect('cache-select', caches, true)
  } catch (e) {
    logLine(`[ERR] refreshLists: ${e}`)
  }
}

function populateSelect(id, items, withNone = false) {
  const sel = el(id)
  if (!sel) {
    logLine(`[WARN] populateSelect: element #${id} not found`)
    return
  }
  sel.innerHTML = ''
  if (withNone) {
    const opt = document.createElement('option')
    opt.value = ''
    opt.textContent = '-- none --'
    sel.appendChild(opt)
  }
  for (const item of items) {
    const opt = document.createElement('option')
    opt.value = item.path
    opt.textContent = `${item.name} (${item.size_mb.toFixed(1)} MB)`
    sel.appendChild(opt)
  }
  logLine(`[INFO] populated #${id} with ${items.length} items`)
}

function populateDatasetList(datasets) {
  const container = el('dataset-list')
  if (!container) return
  container.innerHTML = ''
  datasets.forEach((ds, idx) => {
    const label = document.createElement('label')
    label.className = 'dataset-item'
    const cb = document.createElement('input')
    cb.type = 'checkbox'
    cb.value = String(idx + 1)
    cb.checked = true
    label.appendChild(cb)
    const lines = ds.lines != null ? ` | ${ds.lines.toLocaleString()} lines` : ''
    label.appendChild(document.createTextNode(` ${idx + 1}. ${ds.name} (${ds.size_mb.toFixed(1)} MB${lines})`))
    container.appendChild(label)
  })
}

function logLine(line) {
  if (!processLogElement) return
  processLogElement.textContent += line + '\n'
  processLogElement.scrollTop = processLogElement.scrollHeight
}

function clearLog() {
  if (processLogElement) processLogElement.textContent = ''
}

async function startTraining() {
  const mode = el('train-mode').value
  const datasetChecks = document.querySelectorAll('#dataset-list input:checked')
  const datasets = datasetChecks.length === el('dataset-list').querySelectorAll('input').length
    ? 'all'
    : Array.from(datasetChecks).map(cb => cb.value).join(' ')

  const maxSeqs = el('train-max-seqs').value
  const lr = el('train-lr').value
  const warmup = el('train-warmup').value
  const epochs = el('train-epochs').value
  const useExistingCache = el('use-existing-cache').checked

  clearLog()
  logLine('[INFO] Starting training...')

  try {
    const id = await invoke('start_training', {
      req: {
        mode,
        datasets,
        max_seqs: maxSeqs || null,
        lr: lr || null,
        warmup: warmup || null,
        epochs: epochs || null,
        use_existing_cache: useExistingCache,
      }
    })
    currentProcessId = id
    logLine(`[INFO] Process started: ${id}`)
  } catch (e) {
    logLine(`[ERR] start_training: ${e}`)
  }
}

async function stopTraining() {
  if (!currentProcessId) return
  try {
    await invoke('stop_training', { id: currentProcessId })
    logLine(`[INFO] Stopped ${currentProcessId}`)
  } catch (e) {
    logLine(`[ERR] stop_training: ${e}`)
  }
}

async function startInference() {
  const mode = el('inference-mode').value
  const weights = el('weights-select').value
  const prompt = el('inference-prompt').value
  if (!weights) {
    logLine('[ERR] Select weights file')
    return
  }
  clearLog()
  logLine('[INFO] Starting inference...')
  try {
    const id = await invoke('start_inference', { mode, weights, prompt })
    currentProcessId = id
    logLine(`[INFO] Process started: ${id}`)
  } catch (e) {
    logLine(`[ERR] start_inference: ${e}`)
  }
}

async function deleteAllGguf() {
  if (!confirm('Delete all .gguf files?')) return
  try {
    await invoke('delete_all_gguf')
    await refreshLists()
    logLine('[INFO] All .gguf deleted')
  } catch (e) {
    logLine(`[ERR] delete_all_gguf: ${e}`)
  }
}

async function deleteAllCache() {
  if (!confirm('Delete all sequence caches?')) return
  try {
    await invoke('delete_all_cache')
    await refreshLists()
    logLine('[INFO] All caches deleted')
  } catch (e) {
    logLine(`[ERR] delete_all_cache: ${e}`)
  }
}

function render() {
  document.querySelector('#app').innerHTML = `
    <aside id="sidebar">
      <h1>ARIA</h1>
      <nav>
        <button data-tab="train" class="active">Training</button>
        <button data-tab="inference">Inference</button>
        <button data-tab="chat">Chat</button>
        <button data-tab="tools">Tools</button>
      </nav>
    </aside>
    <main>
      <section id="tab-train" class="tab active">
        <h2>Training</h2>
        <div class="grid">
          <label>Mode
            <select id="train-mode">
              <option value="fresh">Train Fresh</option>
              <option value="debug">Debug Training</option>
              <option value="sft">SFT Train</option>
              <option value="tiny">Tiny Train</option>
            </select>
          </label>
          <label>Max Seqs
            <input id="train-max-seqs" type="text" placeholder="all" />
          </label>
          <label>LR
            <input id="train-lr" type="text" value="0.0001" />
          </label>
          <label>Warmup
            <input id="train-warmup" type="text" value="5000" />
          </label>
          <label>Epochs
            <input id="train-epochs" type="text" value="2" />
          </label>
        </div>
        <label class="check">
          <input id="use-existing-cache" type="checkbox" checked />
          Use existing cache
        </label>
        <h3>Datasets</h3>
        <div id="dataset-list" class="list-box"></div>
        <div class="buttons">
          <button id="btn-start-train">Start Training</button>
          <button id="btn-stop-train" class="danger">Stop</button>
        </div>
      </section>

      <section id="tab-inference" class="tab">
        <h2>Inference</h2>
        <label>Weights
          <select id="weights-select"></select>
        </label>
          <label>Mode
          <select id="inference-mode">
            <option value="greedy">Greedy</option>
            <option value="sample">Sample</option>
            <option value="inference">Inference</option>
            <option value="test_suite">Test Suite</option>
            <option value="debug_logits">Debug Logits</option>
          </select>
        </label>
        <label>Prompt
          <textarea id="inference-prompt" rows="4" placeholder="Enter prompt..."></textarea>
        </label>
        <div class="buttons">
          <button id="btn-start-inference">Run Inference</button>
        </div>
      </section>

      <section id="tab-chat" class="tab">
        <h2>Chat</h2>
        <p class="hint">Not yet implemented. Chat will run interactive inference with persistent stdin.</p>
        <div id="chat-history" class="chat-box"></div>
        <div class="chat-input-row">
          <input id="chat-input" type="text" placeholder="Type message..." disabled />
          <button id="btn-send" disabled>Send</button>
        </div>
      </section>

      <section id="tab-tools" class="tab">
        <h2>Tools</h2>
        <div class="buttons">
          <button id="btn-delete-gguf" class="danger">Delete all .gguf</button>
          <button id="btn-delete-cache" class="danger">Delete all caches</button>
          <button id="btn-refresh">Refresh lists</button>
        </div>
        <label>Cache for training
          <select id="cache-select"></select>
        </label>
        <h3>Export Q4_0</h3>
        <div class="grid">
          <label>Source
            <input id="export-source" type="text" value="aria_checkpoint.gguf" />
          </label>
          <label>Target
            <input id="export-target" type="text" value="aria_inference.gguf" />
          </label>
        </div>
        <div class="buttons">
          <button id="btn-export">Export GGUF</button>
        </div>
      </section>

      <section id="console-section">
        <h2>Console</h2>
        <pre id="process-log"></pre>
      </section>
    </main>
  `
}

function bindTabs() {
  document.querySelectorAll('#sidebar button[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#sidebar button').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'))
      el(`tab-${btn.dataset.tab}`).classList.add('active')
    })
  })
}

function bindActions() {
  el('btn-start-train').addEventListener('click', startTraining)
  el('btn-stop-train').addEventListener('click', stopTraining)
  el('btn-start-inference').addEventListener('click', startInference)
  el('btn-delete-gguf').addEventListener('click', deleteAllGguf)
  el('btn-delete-cache').addEventListener('click', deleteAllCache)
  el('btn-refresh').addEventListener('click', refreshLists)
  el('btn-export').addEventListener('click', exportGguf)

  el('btn-send').addEventListener('click', () => {
    const input = el('chat-input')
    const hist = el('chat-history')
    if (!input.value.trim()) return
    hist.innerHTML += `<div class="msg user">${escapeHtml(input.value)}</div>`
    input.value = ''
    hist.scrollTop = hist.scrollHeight
  })
}

async function exportGguf() {
  const source = el('export-source').value
  const target = el('export-target').value
  clearLog()
  logLine('[INFO] Starting export...')
  try {
    const id = await invoke('export_gguf', { source, target })
    currentProcessId = id
    logLine(`[INFO] Export started: ${id}`)
  } catch (e) {
    logLine(`[ERR] export_gguf: ${e}`)
  }
}

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

async function init() {
  render()
  processLogElement = el('process-log')
  bindTabs()
  bindActions()
  await refreshLists()

  await listen('process-log', event => {
    const { line } = event.payload
    logLine(line)
  })
}

init().catch(e => console.error(e))
