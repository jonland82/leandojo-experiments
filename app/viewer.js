/* Dependency-free 3D point-cloud viewer for LeanDojo proof topics.
   Perspective projection + painter's algorithm on a 2D canvas; no WebGL,
   no external libraries, runs straight off file://. */
(function () {
  "use strict";

  var D = window.PROOF_DATA;
  var PTS = D.points, VIEWS = D.views;
  var colorView = "style", CL = [], K = 0;

  // ------------------------------------------------------------- palette
  // Evenly spaced hues with alternating lightness so adjacent topics stay
  // distinguishable even at k = 17.
  function palette(n) {
    var out = [];
    for (var i = 0; i < n; i++) {
      var h = (i * 360 / n + (i % 2) * 18) % 360;
      var l = i % 3 === 0 ? 68 : (i % 3 === 1 ? 58 : 76);
      out.push("hsl(" + h.toFixed(1) + " 68% " + l + "%)");
    }
    return out;
  }
  var COLORS = [], LABEL = [];

  // --------------------------------------------------------------- state
  var layout = "tsne";
  var on = [];
  var sizeBy = true, fog = true;
  var query = "";
  var selected = -1, hover = -1;

  function topicSlot(p) {
    var topicId = colorView === "style" ? p.c : p.domain_c;
    return topicId >= 0 ? topicId : K;
  }

  var cam = { yaw: 0.6, pitch: 0.35, dist: 190, panX: 0, panY: 0, fovK: 1.0 };

  var cv = document.getElementById("cv");
  var ctx = cv.getContext("2d");
  var tip = document.getElementById("tip");
  var hud = document.getElementById("hud");
  var W = 0, H = 0, DPR = 1;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    draw();
  }
  window.addEventListener("resize", resize);

  // ---------------------------------------------------------- projection
  // Rotate world by yaw (about Y) then pitch (about X), translate along -Z by
  // cam.dist, then perspective-divide.
  var proj = new Float32Array(PTS.length * 3);   // sx, sy, depth
  var order = new Int32Array(PTS.length);
  var vis = new Uint8Array(PTS.length);

  function project() {
    var cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    var cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    var f = (H * 0.9) * cam.fovK;
    var cx = W / 2 + cam.panX, cyy = H / 2 + cam.panY;

    for (var i = 0; i < PTS.length; i++) {
      var p = PTS[i][layout];
      var x = p[0], y = p[1], z = p[2];
      var x1 = cy * x + sy * z;
      var z1 = -sy * x + cy * z;
      var y2 = cp * y - sp * z1;
      var z2 = sp * y + cp * z1 + cam.dist;
      if (z2 < 1) z2 = 1;
      var s = f / z2;
      proj[i * 3] = cx + x1 * s;
      proj[i * 3 + 1] = cyy - y2 * s;
      proj[i * 3 + 2] = z2;
    }
  }

  function matches(p) {
    if (!query) return true;
    return (p.name + " " + p.file).toLowerCase().indexOf(query) !== -1;
  }

  function computeVisible() {
    var n = 0;
    for (var i = 0; i < PTS.length; i++) {
      var v = on[topicSlot(PTS[i])] && matches(PTS[i]) ? 1 : 0;
      vis[i] = v;
      if (v) order[n++] = i;
    }
    // painter's algorithm: far points first
    var sub = Array.prototype.slice.call(order.subarray(0, n));
    sub.sort(function (a, b) { return proj[b * 3 + 2] - proj[a * 3 + 2]; });
    for (var j = 0; j < n; j++) order[j] = sub[j];
    return n;
  }

  function radius(p, depth) {
    var base = sizeBy ? 2.0 + Math.sqrt(p.n) * 1.15 : 3.4;
    return Math.max(1, base * (200 / depth));
  }

  function draw() {
    project();
    var n = computeVisible();

    ctx.clearRect(0, 0, W, H);
    var g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#0d1119"); g.addColorStop(1, "#080a10");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

    var dmin = Infinity, dmax = -Infinity;
    for (var j = 0; j < n; j++) {
      var d = proj[order[j] * 3 + 2];
      if (d < dmin) dmin = d;
      if (d > dmax) dmax = d;
    }
    var span = Math.max(dmax - dmin, 1e-6);

    for (var k = 0; k < n; k++) {
      var i = order[k];
      var p = PTS[i];
      var sx = proj[i * 3], sy = proj[i * 3 + 1], dz = proj[i * 3 + 2];
      if (sx < -40 || sx > W + 40 || sy < -40 || sy > H + 40) continue;
      var r = radius(p, dz);
      var a = fog ? 0.30 + 0.70 * (1 - (dz - dmin) / span) : 0.85;
      if (i === selected || i === hover) a = 1;

      ctx.globalAlpha = a;
      ctx.fillStyle = COLORS[topicSlot(p)];
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, 6.283185);
      ctx.fill();

      if (i === selected) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(sx, sy, r + 4.5, 0, 6.283185); ctx.stroke();
      } else if (i === hover) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#ffffffaa"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(sx, sy, r + 3, 0, 6.283185); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    hud.textContent = n + " / " + PTS.length + " shown  ·  " + colorView + " topics  ·  " + layout.toUpperCase() +
      "  ·  yaw " + (cam.yaw * 57.3).toFixed(0) + "°  pitch " + (cam.pitch * 57.3).toFixed(0) + "°";
  }

  function pick(mx, my) {
    var best = -1, bestD = 14 * 14;
    for (var j = 0; j < PTS.length; j++) {
      if (!vis[j]) continue;
      var dx = proj[j * 3] - mx, dy = proj[j * 3 + 1] - my;
      var d2 = dx * dx + dy * dy;
      var r = radius(PTS[j], proj[j * 3 + 2]);
      var lim = Math.max(r + 3, 5); lim *= lim;
      if (d2 < lim && d2 < bestD) { bestD = d2; best = j; }
    }
    return best;
  }

  // ------------------------------------------------------------- controls
  var dragging = false, panning = false, lx = 0, ly = 0, moved = 0;

  cv.addEventListener("mousedown", function (e) {
    dragging = true; panning = e.shiftKey || e.button === 1;
    lx = e.offsetX; ly = e.offsetY; moved = 0;
    cv.classList.add("drag");
  });
  window.addEventListener("mouseup", function () {
    dragging = false; cv.classList.remove("drag");
  });
  cv.addEventListener("mousemove", function (e) {
    var mx = e.offsetX, my = e.offsetY;
    if (dragging) {
      var dx = mx - lx, dy = my - ly;
      moved += Math.abs(dx) + Math.abs(dy);
      if (panning) { cam.panX += dx; cam.panY += dy; }
      else {
        cam.yaw += dx * 0.006;
        cam.pitch += dy * 0.006;
        cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch));
      }
      lx = mx; ly = my;
      tip.hidden = true;
      draw();
      return;
    }
    var h = pick(mx, my);
    if (h !== hover) { hover = h; draw(); }
    if (h >= 0) {
      var p = PTS[h];
      tip.hidden = false;
      tip.style.left = mx + "px"; tip.style.top = my + "px";
      var slot = topicSlot(p);
      tip.innerHTML = "<b>" + esc(p.name) + "</b><br><span class='f'>" + esc(p.file) +
        "</span><br><span class='f'>" + p.n + " tactics · " + esc(LABEL[slot]) + "</span>";
    } else {
      tip.hidden = true;
    }
  });
  cv.addEventListener("mouseleave", function () {
    tip.hidden = true; hover = -1; draw();
  });
  cv.addEventListener("click", function (e) {
    if (moved > 5) return;
    var h = pick(e.offsetX, e.offsetY);
    selected = h;
    showDetail(h);
    draw();
  });
  cv.addEventListener("wheel", function (e) {
    e.preventDefault();
    cam.dist *= Math.exp(e.deltaY * 0.0012);
    cam.dist = Math.max(20, Math.min(1200, cam.dist));
    draw();
  }, { passive: false });

  // --------------------------------------------------------------- panels
  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  }

  var detail = document.getElementById("detail");
  function topicTag(viewName, topicId, weight) {
    if (topicId < 0) {
      var missing = viewName === "domain" ? "no explicit premise signal" : "no modeled style weight";
      return "<span class='tag'>" + missing + "</span>";
    }
    var view = VIEWS[viewName], topic = view.topics[topicId];
    var color = palette(view.k)[topicId];
    var suffix = weight === undefined ? "" : " · weight " + weight.toFixed(2);
    return "<span class='tag' style='background:" + color + "22;color:" + color + "'>" +
      esc(viewName + ": " + topic.label + suffix) + "</span>";
  }

  function showDetail(i) {
    if (i < 0) {
      detail.innerHTML = "<p class='hint'>Drag to orbit · scroll to zoom · " +
        "shift-drag to pan · click a point for its proof.</p>";
      return;
    }
    var p = PTS[i];
    var styleWeight = p.style_mix.length ? p.style_mix[0][1] : 0;
    var domainWeight = p.domain_mix.length ? p.domain_mix[0][1] : undefined;
    detail.innerHTML =
      "<h2>" + esc(p.name) + "</h2>" +
      "<p class='meta'>" + esc(p.file) + "<br>" + p.n + " tactic step" +
      (p.n === 1 ? "" : "s") + "</p>" +
      topicTag("style", p.c, styleWeight) +
      topicTag("domain", p.domain_c, domainWeight) +
      "<p class='meta'>mixture entropy: style " + p.style_entropy.toFixed(2) +
      " · domain " + p.domain_entropy.toFixed(2) + "</p>" +
      "<pre>" + esc(p.script) + "</pre>";
  }

  var legend = document.getElementById("legend");
  function rebuildLegend() {
    legend.innerHTML = "";
    CL.forEach(function (topic, slot) {
      var li = document.createElement("li");
      li.innerHTML = "<span class='sw' style='background:" + COLORS[slot] + "'></span>" +
        "<span class='nm' title='" + esc((topic.top_terms || []).join(', ')) + "'>" +
        esc(LABEL[slot]) + "</span><span class='ct'>" + topic.size + "</span>";
      li.addEventListener("click", function () {
        on[slot] = !on[slot];
        li.classList.toggle("off", !on[slot]);
        draw();
      });
      legend.appendChild(li);
    });
  }

  function setColorView(name) {
    colorView = name;
    K = VIEWS[name].k;
    CL = VIEWS[name].topics.slice();
    if (VIEWS[name].unclassified) {
      CL.push({
        id: K, size: VIEWS[name].unclassified,
        label: name === "domain" ? "no explicit premise signal" : "no modeled topic weight",
        top_terms: []
      });
    }
    COLORS = palette(CL.length);
    if (CL.length > K) COLORS[K] = "#667085";
    LABEL = CL.map(function (topic) { return topic.label; });
    on = CL.map(function () { return true; });
    document.getElementById("kval").textContent = K;
    rebuildLegend();
    if (selected >= 0) showDetail(selected);
  }

  document.getElementById("alltoggle").addEventListener("click", function () {
    var anyOn = on.some(Boolean);
    for (var i = 0; i < on.length; i++) on[i] = !anyOn;
    Array.prototype.forEach.call(legend.children, function (li, i) {
      li.classList.toggle("off", !on[i]);
    });
    this.textContent = anyOn ? "all" : "none";
    draw();
  });

  Array.prototype.forEach.call(document.querySelectorAll(".segbtn[data-layout]"), function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll(".segbtn[data-layout]").forEach(function (x) {
        x.classList.remove("active");
      });
      b.classList.add("active");
      layout = b.dataset.layout;
      draw();
    });
  });

  Array.prototype.forEach.call(document.querySelectorAll(".viewbtn"), function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll(".viewbtn").forEach(function (x) {
        x.classList.remove("active");
      });
      b.classList.add("active");
      setColorView(b.dataset.view);
      draw();
    });
  });

  document.getElementById("sizeby").addEventListener("change", function () {
    sizeBy = this.checked; draw();
  });
  document.getElementById("fog").addEventListener("change", function () {
    fog = this.checked; draw();
  });
  document.getElementById("search").addEventListener("input", function () {
    query = this.value.trim().toLowerCase(); draw();
  });

  // ------------------------------------------------------------------ go
  document.getElementById("npts").textContent = PTS.length;
  setColorView("style");
  resize();

  // gentle idle auto-rotation until the user first interacts
  var idle = true;
  cv.addEventListener("mousedown", function () { idle = false; });
  cv.addEventListener("wheel", function () { idle = false; });
  (function spin() {
    if (idle) { cam.yaw += 0.0022; draw(); }
    requestAnimationFrame(spin);
  })();
})();
