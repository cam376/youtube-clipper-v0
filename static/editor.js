// Caption editor, find & replace, glossary.
(function () {
  const K = window.Kivro;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------- caption editor
  const modal = $("modal-captions"), cuesBox = $("cues");
  let editing = null;

  K.openCaptionEditor = function (clip) {
    editing = clip;
    $("captions-title").textContent = `Edit captions · ${clip.label}`;
    cuesBox.innerHTML = "";
    clip.cues.forEach((cue, i) => {
      const row = document.createElement("div");
      row.className = "cue";
      row.innerHTML = `<span class="time">${fmtCue(cue.start)} → ${fmtCue(cue.end)}</span><input type="text" data-i="${i}" value="${K.escapeHtml(cue.text)}">`;
      const input = row.querySelector("input");
      input.addEventListener("input", () => input.classList.toggle("changed", input.value.trim() !== cue.text.trim()));
      cuesBox.appendChild(row);
    });
    modal.classList.remove("hidden");
    const first = cuesBox.querySelector("input");
    if (first) first.focus();
  };

  $("captions-cancel").addEventListener("click", () => modal.classList.add("hidden"));
  $("captions-save").addEventListener("click", async () => {
    if (!editing) return;
    const cues = [...cuesBox.querySelectorAll("input")].map((inp) => ({ text: inp.value }));
    $("captions-save").disabled = true;
    try {
      const res = await K.api("PUT", `api/jobs/${K.currentJobId}/clips/${editing.id}/captions`, { cues });
      modal.classList.add("hidden");
      if (!res.changed_cues) K.showError("No caption text changed.");
      K.poll();
    } catch (err) {
      K.showError(err.message);
    } finally {
      $("captions-save").disabled = false;
    }
  });

  function fmtCue(s) {
    const m = Math.floor(s / 60), r = s - m * 60;
    return `${String(m).padStart(2, "0")}:${r.toFixed(2).padStart(5, "0")}`;
  }

  // ---------------------------------------------------------- find & replace
  const fr = $("modal-fr");
  $("btn-findreplace").addEventListener("click", () => {
    $("fr-preview").textContent = "";
    $("fr-apply").disabled = true;
    fr.classList.remove("hidden");
    $("fr-find").focus();
  });
  $("fr-cancel").addEventListener("click", () => fr.classList.add("hidden"));
  $("fr-find").addEventListener("input", () => ($("fr-apply").disabled = true));

  $("fr-check").addEventListener("click", async () => {
    try {
      const res = await K.api("POST", `api/jobs/${K.currentJobId}/find-replace`, {
        find: $("fr-find").value, replace: $("fr-replace").value, apply: false, case_sensitive: $("fr-case").checked,
      });
      if (!res.matches.length) {
        $("fr-preview").textContent = "No clip contains this text.";
        $("fr-apply").disabled = true;
        return;
      }
      $("fr-preview").textContent = `${res.affected_clips} clip(s) will be rerendered:\n` +
        res.matches.map((m) => `• ${m.label}: ${m.cue_count} cue(s) — e.g. "${m.examples[0]}"`).join("\n");
      $("fr-apply").disabled = false;
    } catch (err) { K.showError(err.message); }
  });

  $("fr-apply").addEventListener("click", async () => {
    $("fr-apply").disabled = true;
    try {
      const res = await K.api("POST", `api/jobs/${K.currentJobId}/find-replace`, {
        find: $("fr-find").value, replace: $("fr-replace").value, apply: true, case_sensitive: $("fr-case").checked,
      });
      $("fr-preview").textContent = `Replaced ${res.replaced} occurrence(s) in ${res.affected_clips} clip(s). Rerendering…`;
      K.poll();
    } catch (err) { K.showError(err.message); }
  });

  // ---------------------------------------------------------- glossary
  const gl = $("modal-glossary");
  $("btn-glossary").addEventListener("click", () => {
    $("glossary-text").value = ((K.job && K.job.glossary) || []).join("\n");
    gl.classList.remove("hidden");
  });
  $("glossary-cancel").addEventListener("click", () => gl.classList.add("hidden"));
  $("glossary-save").addEventListener("click", async () => {
    const terms = $("glossary-text").value.split("\n").map((t) => t.trim()).filter(Boolean);
    try {
      await K.api("PUT", `api/jobs/${K.currentJobId}/glossary`, { terms });
      gl.classList.add("hidden");
      K.poll();
    } catch (err) { K.showError(err.message); }
  });
})();
