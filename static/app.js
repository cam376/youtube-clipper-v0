const form = document.getElementById("form");
const urlInput = document.getElementById("url");
const button = document.getElementById("generate");
const statusBox = document.getElementById("status");
const statusText = document.getElementById("status-text");
const statusDetail = document.getElementById("status-detail");
const errorBox = document.getElementById("error");
const clipsBox = document.getElementById("clips");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resetUI();
  button.disabled = true;

  let jobId;
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlInput.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    jobId = data.job_id;
  } catch (err) {
    showError(err.message);
    button.disabled = false;
    return;
  }

  showStatus("Importing video...");
  poll(jobId);
});

async function poll(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || "Job lookup failed");

    showStatus(job.status, job.detail);

    if (job.stage === "done") {
      statusBox.classList.add("done");
      if (job.ranking_note) statusDetail.textContent = job.ranking_note;
      renderClips(job.clips);
      button.disabled = false;
      return;
    }
    if (job.stage === "error") {
      statusBox.classList.add("hidden");
      showError(job.error);
      button.disabled = false;
      return;
    }
  } catch (err) {
    showError(err.message);
    button.disabled = false;
    return;
  }
  setTimeout(() => poll(jobId), 2000);
}

function renderClips(clips) {
  clipsBox.innerHTML = "";
  clips.forEach((clip) => {
    const card = document.createElement("div");
    card.className = "clip";
    card.innerHTML = `
      <h2>Clip ${clip.index}</h2>
      <video controls preload="metadata" src="${clip.url}"></video>
      <div class="meta">${fmt(clip.start)} – ${fmt(clip.end)} · ${clip.duration}s</div>
      <div class="text">${escapeHtml(clip.text)}</div>
      <a class="download" href="${clip.url}?download=true" download="${clip.filename}">Download</a>
    `;
    clipsBox.appendChild(card);
  });
}

function showStatus(text, detail = "") {
  statusBox.classList.remove("hidden", "done");
  statusText.textContent = text;
  statusDetail.textContent = detail;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function resetUI() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
  clipsBox.innerHTML = "";
  statusBox.classList.add("hidden");
}

function fmt(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
