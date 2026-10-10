// Client library export modal.
(function () {
  const K = window.Kivro;
  const $ = (id) => document.getElementById(id);
  const modal = $("modal-export");
  let pollTimer = null;

  $("btn-export").addEventListener("click", () => {
    const n = (K.job && K.job.library_count) || 0;
    $("export-count").textContent = n ? `${n} clip${n === 1 ? "" : "s"} marked for the client library.` : "No clip is marked yet. Use \"Add to client library\" on the clips you want to show.";
    $("export-run").disabled = n === 0;
    $("ex-watermark").checked = K.options ? !!K.options.preview_watermark_default : true;
    $("export-progress").textContent = "";
    $("export-result").classList.add("hidden");
    if (!$("ex-title").value && $("ex-client").value) $("ex-title").value = `${$("ex-client").value}'s Content Library`;
    modal.classList.remove("hidden");
  });
  $("ex-client").addEventListener("input", () => {
    const c = $("ex-client").value.trim();
    if (c && (!$("ex-title").dataset.touched)) $("ex-title").value = `${c}'s Content Library`;
  });
  $("ex-title").addEventListener("input", () => ($("ex-title").dataset.touched = "1"));
  $("ex-showprice").addEventListener("change", () => $("ex-price-row").classList.toggle("hidden", !$("ex-showprice").checked));
  $("export-cancel").addEventListener("click", () => { clearTimeout(pollTimer); modal.classList.add("hidden"); });

  $("export-run").addEventListener("click", async () => {
    $("export-run").disabled = true;
    $("export-progress").textContent = "Starting…";
    try {
      const st = await K.api("POST", `api/jobs/${K.currentJobId}/export-library`, {
        client_name: $("ex-client").value,
        title: $("ex-title").value || null,
        show_price: $("ex-showprice").checked,
        price_per_clip: $("ex-showprice").checked && $("ex-price").value ? parseFloat($("ex-price").value) : null,
        currency: "EUR",
        whatsapp_number: $("ex-whatsapp").value || null,
        watermark: $("ex-watermark").checked,
      });
      await track(st.export_id);
    } catch (err) {
      $("export-progress").textContent = "";
      K.showError(err.message);
      $("export-run").disabled = false;
    }
  });

  async function track(exportId) {
    const st = await K.api("GET", `api/exports/${exportId}`);
    if (st.stage === "exporting") {
      $("export-progress").textContent = `Exporting… ${st.detail || ""}`;
      pollTimer = setTimeout(() => track(exportId), 1500);
      return;
    }
    $("export-run").disabled = false;
    if (st.stage === "error") {
      $("export-progress").textContent = "";
      K.showError(st.error);
      return;
    }
    const lib = st.library;
    $("export-progress").textContent = "Done.";
    const r = $("export-result");
    r.classList.remove("hidden");
    r.innerHTML = `
      <div><b>${K.escapeHtml(lib.title)}</b> — ${lib.clip_count} clips</div>
      <div>Folder: <code>${K.escapeHtml(lib.path)}</code></div>
      <div>Preview now: <a href="${st.preview_url}" target="_blank" rel="noopener">${st.preview_url}</a></div>
      <div>Standalone preview: <code>python -m http.server 8080 --directory "${K.escapeHtml(lib.path)}"</code> then open http://localhost:8080</div>
      <div>Upload the whole folder to any static host (Netlify Drop, Cloudflare Pages, GitHub Pages, S3) and send the link.</div>`;
    K.poll();
  }
})();
