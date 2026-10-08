// Kivro review UI: jobs list, pipeline status, clip cards.
// editor.js adds the caption editor / find & replace / glossary,
// export.js adds the client library export. All share window.Kivro.

const Kivro = window.Kivro = {
  options: null,
  currentJobId: null,
  job: null,
  pollTimer: null,
  api,
  loadJob,
  renderJob,
  escapeHtml,
  fmt,
};

const $ = (id) => document.getElementById(id);
const form = $("form"), urlInput = $("url"), button = $("generate");
const statusBox = $("status"), statusText = $("status-text"), statusDetail = $("status-detail");
const errorBox = $("error"), jobsList = $("jobs-list"), jobSection = $("job"), clipsBox = $("clips");

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${method} ${path} failed (${res.status})`);
  return data;
}

// ---------------------------------------------------------------- generate
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  button.disabled = true;
  try {
    const { job_id } = await api("POST", "/api/generate", { url: urlInput.value });
    showStatus("Waiting for a free slot...");
    await loadJobs();
    await loadJob(job_id);
  } catch (err) {
    showError(err.message);
    button.disabled = false;
  }
});

// ---------------------------------------------------------------- jobs list
async function loadJobs() {
  const jobs = await api("GET", "/api/jobs");
  jobsList.innerHTML = "";
  if (!jobs.length) {
    jobsList.innerHTML = '<li class="m">No jobs yet.</li>';
    return;
  }
  jobs.forEach((j) => {
    const li = document.createElement("li");
    li.className = j.job_id === Kivro.currentJobId ? "active" : "";
    const state = j.stage === "done" ? `${j.clip_count} clips · ${j.library_count} in library` : (j.status || j.stage);
    li.innerHTML = `<span class="t">${escapeHtml(j.title || j.url || j.job_id)}</span><span class="m">${escapeHtml(state)} · ${escapeHtml(j.created_at || "")}</span>`;
    li.addEventListener("click", () => loadJob(j.job_id));
    jobsList.appendChild(li);
  });
}

// ---------------------------------------------------------------- one job
async function loadJob(jobId) {
  Kivro.currentJobId = jobId;
  clearTimeout(Kivro.pollTimer);
  await poll();
}

async function poll() {
  const jobId = Kivro.currentJobId;
  if (!jobId) return;
  let job;
  try {
    job = await api("GET", `/api/jobs/${jobId}`);
  } catch (err) {
    showError(err.message);
    return;
  }
  Kivro.job = job;
  renderJob(job);
  const busy = job.stage !== "done" && job.stage !== "error";
  const rendering = (job.clips || []).some((c) => c.render_state === "queued" || c.render_state === "rendering");
  if (busy || rendering) {
    Kivro.pollTimer = setTimeout(poll, 2000);
  } else {
    button.disabled = false;
  }
}

function renderJob(job) {
  // status line
  if (job.stage === "done") {
    showStatus(`${job.clip_count ?? (job.clips || []).length} clips ready`, [job.transcription_note, job.ranking_note].filter(Boolean).join(" · "));
    statusBox.classList.add("done");
  } else if (job.stage === "error") {
    statusBox.classList.add("hidden");
    showError(job.error);
  } else {
    showStatus(job.status, job.detail);
  }

  jobSection.classList.remove("hidden");
  $("job-title").textContent = (job.source && job.source.title) || job.url || job.job_id;
  $("job-meta").textContent = `${job.job_id} · ${(job.source && job.source.url) || job.url || ""}`;

  const sel = job.selection || {};
  const rk = job.ranking || {};
  $("job-summary").innerHTML = [
    `<span><b>${(job.clips || []).length}</b> clips</span>`,
    `<span><b>${job.library_count || 0}</b> in client library</span>`,
    rk.candidates ? `<span><b>${rk.candidates}</b> candidates scored, max ${rk.score_max ?? "?"}, median ${rk.score_median ?? "?"}</span>` : "",
    sel.min_clip_score != null ? `<span>threshold <b>${sel.min_clip_score}</b></span>` : "",
    (job.glossary || []).length ? `<span>glossary: <b>${job.glossary.length}</b> terms</span>` : "",
  ].join("");

  renderClips(job.clips || []);
  loadJobs().catch(() => {});
}

function renderClips(clips) {
  clipsBox.innerHTML = "";
  const styles = (Kivro.options && Kivro.options.styles) || [];
  const fonts = (Kivro.options && Kivro.options.fonts) || [];
  clips.forEach((clip) => {
    const card = document.createElement("div");
    card.className = "clip" + (clip.in_library ? " in-library" : "");
    card.dataset.id = clip.id;
    const busy = clip.render_state === "queued" || clip.render_state === "rendering";
    const badges = [
      clip.edited ? '<span class="badge edited">edited</span>' : "",
      clip.in_library ? '<span class="badge lib">in client library</span>' : "",
      busy ? `<span class="badge rendering">${clip.render_state}…</span>` : "",
      clip.render_state === "error" ? `<span class="badge warn" title="${escapeHtml(clip.render_error || "")}">render error</span>` : "",
      clip.font_note ? `<span class="badge warn" title="${escapeHtml(clip.font_note)}">font fallback</span>` : "",
    ].join("");
    const styleOpts = styles.map((s) => `<option value="${s.id}" ${s.id === clip.style ? "selected" : ""}>${escapeHtml(s.label)}</option>`).join("");
    const fontOpts = fonts.map((f) => `<option value="${f.id}" ${f.id === clip.font ? "selected" : ""}>${escapeHtml(f.label)}${f.fallback ? " (fallback)" : ""}</option>`).join("");
    card.innerHTML = `
      <video controls preload="metadata" src="${clip.url}"></video>
      <div class="head"><h3>${escapeHtml(clip.label)}</h3><span class="score">Score ${clip.score != null ? Math.round(clip.score * 10) : "–"}</span></div>
      <div class="meta">${fmt(clip.start)} – ${fmt(clip.end)} · ${fmtDur(clip.duration)} · ${layoutLabel(clip.layout)}<br>
        Style: <b>${escapeHtml(styleLabel(clip.style))}</b> · Font: <b>${escapeHtml(fontLabel(clip.font))}</b></div>
      <div class="badges">${badges}</div>
      <div class="text">${escapeHtml(clip.text || "")}</div>
      <div class="row">
        <select class="sel-style" title="Subtitle style">${styleOpts}</select>
        <select class="sel-font" title="Font">${fontOpts}</select>
        <button class="small secondary btn-apply" title="Apply style/font and rerender this clip">Apply</button>
      </div>
      <div class="actions">
        <button class="small secondary btn-edit">Edit captions</button>
        <button class="small toggle btn-lib ${clip.in_library ? "on" : ""}">${clip.in_library ? "In client library ✓" : "Add to client library"}</button>
        <a class="download" href="${clip.url.split("?")[0]}?download=true" download="${clip.filename}">Download HD</a>
        <button class="small secondary btn-rerender" title="Rerender with current captions/style">Rerender</button>
      </div>
    `;
    if (busy) card.querySelectorAll("button").forEach((b) => (b.disabled = true));
    card.querySelector(".btn-apply").addEventListener("click", () => applyStyle(clip, card));
    card.querySelector(".btn-lib").addEventListener("click", () => toggleLibrary(clip));
    card.querySelector(".btn-edit").addEventListener("click", () => Kivro.openCaptionEditor && Kivro.openCaptionEditor(clip));
    card.querySelector(".btn-rerender").addEventListener("click", () => rerender(clip));
    clipsBox.appendChild(card);
  });
}

async function applyStyle(clip, card) {
  const style = card.querySelector(".sel-style").value;
  const font = card.querySelector(".sel-font").value;
  try {
    await api("PUT", `/api/jobs/${Kivro.currentJobId}/clips/${clip.id}/style`, { style, font });
    poll();
  } catch (err) { showError(err.message); }
}

async function toggleLibrary(clip) {
  try {
    await api("PUT", `/api/jobs/${Kivro.currentJobId}/clips/${clip.id}/library`, { selected: !clip.in_library });
    poll();
  } catch (err) { showError(err.message); }
}

async function rerender(clip) {
  try {
    await api("POST", `/api/jobs/${Kivro.currentJobId}/clips/${clip.id}/rerender`);
    poll();
  } catch (err) { showError(err.message); }
}

// ---------------------------------------------------------------- helpers
function showStatus(text, detail = "") {
  statusBox.classList.remove("hidden", "done");
  statusText.textContent = text;
  statusDetail.textContent = detail;
}
function showError(message) { errorBox.textContent = message; errorBox.classList.remove("hidden"); }
function hideError() { errorBox.classList.add("hidden"); errorBox.textContent = ""; }
function fmt(seconds) { const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60); return `${m}:${String(s).padStart(2, "0")}`; }
function fmtDur(seconds) { const s = Math.round(seconds || 0); return `00:${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; }
function layoutLabel(layout) { return { TWO_PERSON: "split-screen", SINGLE_PERSON: "face-centred", CENTER_CROP: "centre crop" }[layout] || ""; }
function styleLabel(id) { const s = (Kivro.options?.styles || []).find((x) => x.id === id); return s ? s.label : id || ""; }
function fontLabel(id) { const f = (Kivro.options?.fonts || []).find((x) => x.id === id); return f ? f.label : id || ""; }
function escapeHtml(str) { return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

Kivro.showError = showError;
Kivro.poll = poll;

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    Kivro.options = await api("GET", "/api/options");
  } catch (err) { showError(err.message); }
  await loadJobs().catch((e) => showError(e.message));
  const hash = location.hash.replace("#", "");
  if (hash) loadJob(hash);
})();
