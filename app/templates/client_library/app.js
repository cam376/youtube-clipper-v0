(function () {
  "use strict";

  var lib = JSON.parse(document.getElementById("library-data").textContent);
  var KEY = "kivro-selection-" + lib.library_id;

  var grid = document.getElementById("grid");
  var bar = document.getElementById("bar");
  var countEl = document.getElementById("count");
  var totalEl = document.getElementById("total");
  var modal = document.getElementById("modal");
  var summaryEl = document.getElementById("summary");
  var waLink = document.getElementById("whatsapp");
  var copiedEl = document.getElementById("copied");

  document.getElementById("subtitle").textContent = lib.subtitle;

  var selected = loadSelection();

  function loadSelection() {
    try {
      var raw = localStorage.getItem(KEY);
      var arr = raw ? JSON.parse(raw) : [];
      return arr.filter(function (id) { return lib.clips.some(function (c) { return c.id === id; }); });
    } catch (e) { return []; }
  }
  function saveSelection() {
    try { localStorage.setItem(KEY, JSON.stringify(selected)); } catch (e) { /* private mode */ }
  }
  function isSelected(id) { return selected.indexOf(id) !== -1; }
  function toggle(id) {
    var i = selected.indexOf(id);
    if (i === -1) selected.push(id); else selected.splice(i, 1);
    saveSelection();
    render();
  }

  function fmtDuration(s) {
    if (s == null) return "";
    var m = Math.floor(s / 60), r = Math.round(s % 60);
    return "00:" + String(m).padStart(2, "0") + ":" + String(r).padStart(2, "0");
  }
  function fmtPrice(v) {
    var cur = lib.currency || "EUR";
    var sym = cur === "EUR" ? "€" : cur === "USD" ? "$" : cur === "GBP" ? "£" : cur + " ";
    return sym + (Math.round(v * 100) / 100).toString().replace(/\.0+$/, "");
  }

  function buildSummary(sel) {
    var chosen = lib.clips.filter(function (c) { return sel.indexOf(c.id) !== -1; });
    var lines = ["Hi, here are the clips I'd like:", ""];
    chosen.forEach(function (c) { lines.push(c.label); });
    lines.push("", "Total selected: " + chosen.length);
    if (lib.show_price && lib.price_per_clip != null) {
      lines.push("Total: " + fmtPrice(lib.price_per_clip * chosen.length));
    }
    lines.push("", "Library: " + lib.title + " (" + lib.library_id + ")");
    return lines.join("\n");
  }
  // exposed for tests
  window.kivroBuildSummary = buildSummary;

  function render() {
    grid.innerHTML = "";
    lib.clips.forEach(function (c) {
      var card = document.createElement("article");
      card.className = "card" + (isSelected(c.id) ? " selected" : "");
      card.dataset.id = c.id;
      var price = (lib.show_price && lib.price_per_clip != null) ? '<div class="price">' + fmtPrice(lib.price_per_clip) + ' per clip</div>' : "";
      card.innerHTML =
        '<video controls playsinline preload="metadata" poster="' + c.poster + '" src="' + c.preview + '"></video>' +
        '<div class="card-body">' +
          '<div class="card-head"><h3>' + c.label + '</h3><span class="dur">' + fmtDuration(c.duration) + '</span></div>' +
          price +
          '<button class="btn select" type="button">' + (isSelected(c.id) ? "Selected ✓" : "Select this clip") + '</button>' +
        '</div>';
      card.querySelector(".btn.select").addEventListener("click", function () { toggle(c.id); });
      grid.appendChild(card);
    });
    var n = selected.length;
    countEl.textContent = n + (n === 1 ? " clip selected" : " clips selected");
    totalEl.textContent = (lib.show_price && lib.price_per_clip != null && n) ? "Total " + fmtPrice(lib.price_per_clip * n) : "";
    bar.hidden = n === 0;
  }

  // pause other videos when one starts playing
  grid.addEventListener("play", function (e) {
    var vids = grid.querySelectorAll("video");
    for (var i = 0; i < vids.length; i++) if (vids[i] !== e.target) vids[i].pause();
  }, true);

  document.getElementById("send").addEventListener("click", function () {
    var text = buildSummary(selected);
    summaryEl.value = text;
    copiedEl.hidden = true;
    if (lib.whatsapp_number) {
      waLink.href = "https://wa.me/" + lib.whatsapp_number + "?text=" + encodeURIComponent(text);
      waLink.hidden = false;
    } else {
      waLink.hidden = true;
    }
    modal.hidden = false;
  });
  document.getElementById("close").addEventListener("click", function () { modal.hidden = true; });
  modal.addEventListener("click", function (e) { if (e.target === modal) modal.hidden = true; });
  document.getElementById("copy").addEventListener("click", function () {
    var text = summaryEl.value;
    function done() { copiedEl.hidden = false; }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { summaryEl.select(); document.execCommand("copy"); done(); });
    } else {
      summaryEl.select(); document.execCommand("copy"); done();
    }
  });

  render();
})();
