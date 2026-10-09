(function () {
  "use strict";

  var PAGE = window.PAGE;
  var byId = {};
  PAGE.items.forEach(function (it) { byId[it.id] = it; });

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var videos = [];

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function play(v) {
    if (!v) return;
    v._wantsPlayback = true;
    v.autoplay = !document.hidden;
    if (document.hidden) return;
    v.muted = true;
    var p = v.play();
    if (p && p.catch) p.catch(function (error) {
      // A browser that blocks autoplay still gives the reader a way to start the clip.
      if (error.name === "NotAllowedError" && v._wantsPlayback) {
        v.controls = true;
        v._autoplayBlocked = true;
      }
    });
  }
  function pause(v) {
    if (!v) return;
    v._wantsPlayback = false;
    v.autoplay = false;
    v.pause();
  }
  function resumeVisibleVideos() {
    videos.forEach(function (v) { if (v._wantsPlayback && v.paused) play(v); });
  }
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      videos.forEach(function (v) { v.autoplay = false; v.pause(); });
    } else resumeVisibleVideos();
  });
  window.addEventListener("pageshow", resumeVisibleVideos);
  document.addEventListener("pointerdown", resumeVisibleVideos, { passive: true });
  document.addEventListener("keydown", resumeVisibleVideos);

  // The videos are research content; reduced-motion preferences still apply to decorative animations.
  function autoplayInView(list, threshold) {
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) play(e.target); else pause(e.target); });
      }, { threshold: threshold });
      list.forEach(function (v) { io.observe(v); });
    } else list.forEach(play);
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (attrs[k] == null) return;
      if (k === "text") node.textContent = attrs[k];
      else if (k === "style") node.style.cssText = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (ch) { if (ch) node.appendChild(ch); });
    return node;
  }
  function icon(name) {
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    var use = document.createElementNS(ns, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }

  function thumb(it) { return "static/thumb/" + it.id + ".jpg"; }

  function output(it) {
    return it.wildicon || it.preview;
  }
  function outputVideo(it, preload) {
    var w = output(it);
    var v = el("video", { muted: "", loop: "", playsinline: "", preload: preload || "none", poster: w.poster, src: w.video,
      "data-example": it.id, "aria-label": "Generated video of the " + it.species.toLowerCase() });
    v.muted = v.defaultMuted = true;
    videos.push(v);
    v.addEventListener("loadeddata", function () { if (v._wantsPlayback) play(v); });
    v.addEventListener("playing", function () {
      if (v._autoplayBlocked) {
        v.controls = !!v.closest("#viewer");
        v._autoplayBlocked = false;
      }
    });
    return v;
  }

  // Reference photograph -> WildIcon video (or an empty slot), equal height, each at its own aspect ratio.
  function pair(it, preload, small) {
    var refAr = it.refWidth / it.refHeight;
    var w = output(it);
    var outAr = w ? w.width / w.height : refAr;
    var v = w ? outputVideo(it, preload) : null;
    var out = v
      ? el("div", { class: "panel-out", style: "--ar:" + outAr.toFixed(4) }, [v])
      : el("div", { class: "panel-out pending", style: "--ar:" + outAr.toFixed(4) }, [
          el("div", { class: "slot" }, [icon("play"), el("span", { text: "Video" })])
        ]);
    var node = el("div", { class: "pair", tabindex: "0", role: "button", "aria-label": "Open the " + it.species.toLowerCase() + " example" }, [
      el("div", { class: "panel-ref", style: "--ar:" + refAr.toFixed(4) }, [
        el("span", { class: "tag", text: "Reference" }),
        el("img", { src: small ? thumb(it) : it.reference, alt: it.species + " reference photograph " + (it.referenceId || ""), loading: "lazy" })
      ]),
      out,
      el("span", { class: "join", style: "--join:" + (100 * refAr / (refAr + outAr)).toFixed(2) + "%", "aria-hidden": "true" }, [icon("right")])
    ]);
    return { node: node, video: v };
  }

  function caption(it, clamp) {
    return el("figcaption", { class: "cap" }, [
      el("span", { class: "name", text: it.species }),
      el("span", { class: "reference-id", text: it.referenceId }),
      el("p", { class: "prompt", text: it.prompt, title: clamp ? it.prompt : null })
    ]);
  }

  function openOnActivate(node, it, guard) {
    node.addEventListener("click", function () { if (!guard || guard()) openViewer(it); });
    node.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openViewer(it); } });
  }

  // ---------- hero wall: staggered columns of examples ----------
  // A tile plays its video when there is one (up to HERO_VIDEOS of them, to keep the page light);
  // other tiles show the video's poster, or the reference photograph while no video exists.

  var HERO_VIDEOS = 10;
  var COLS = 7, ROWS = 8;
  var TILE_AR = ["4 / 3", "1 / 1", "4 / 5", "4 / 3", "3 / 2", "1 / 1"];
  var wall = $("[data-wall]");
  var wallVideos = [];
  var heroIds = PAGE.heroWall;
  for (var c = 0; c < COLS; c++) {
    var col = el("div", { class: "wall-col" });
    for (var r = 0; r < ROWS; r++) {
      // rows past the list take tiles from two columns over, so a picture never repeats in its own column
      var k = r * COLS + c;
      if (k >= heroIds.length) k = ((r - Math.floor(heroIds.length / COLS)) * COLS + (c + 2) % COLS) % heroIds.length;
      var it = byId[heroIds[k % heroIds.length]];
      var tile = el("div", { class: "wall-tile", style: "--ar:" + TILE_AR[(r + c) % TILE_AR.length] });
      var w = output(it);
      var firstUse = r * COLS + c < heroIds.length;
      if (w && firstUse && wallVideos.length < HERO_VIDEOS) {
        var v = outputVideo(it, "metadata"); wallVideos.push(v); tile.appendChild(v);
      } else {
        tile.appendChild(el("img", { src: w ? w.poster : thumb(it), alt: "", loading: r > 3 ? "lazy" : "eager" }));
      }
      col.appendChild(tile);
    }
    wall.appendChild(col);
  }
  autoplayInView(wallVideos, 0.1);

  // ---------- back to top ----------

  (function () {
    var btn = $("#to-top");
    var bar = $(".ring .bar", btn);
    var C = 2 * Math.PI * 22;
    var ticking = false;
    function update() {
      ticking = false;
      var max = document.documentElement.scrollHeight - window.innerHeight;
      var k = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      bar.style.strokeDashoffset = String(C * (1 - k));
      btn.classList.toggle("show", window.scrollY > window.innerHeight * 0.8);
    }
    window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    window.addEventListener("resize", update);
    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
      btn.blur();
    });
    update();
  })();

  // ---------- top nav ----------

  var nav = $("#topnav");
  var hero = $(".hero");
  function onScroll() { nav.classList.toggle("show", window.scrollY > hero.offsetHeight - 120); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if ("IntersectionObserver" in window) {
    var links = {};
    $$(".nav-links a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var a = links[e.target.id];
        if (a && e.isIntersecting) { $$(".nav-links a").forEach(function (x) { x.classList.remove("active"); }); a.classList.add("active"); }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    $$("main > section[id]").forEach(function (s) { spy.observe(s); });
  }

  // ---------- abstract ----------

  var abs = $("#abstract-body");
  $("#abstract-toggle").addEventListener("click", function () {
    var open = abs.classList.toggle("open");
    this.setAttribute("aria-expanded", String(open));
    this.textContent = open ? "Show less" : "Read the full abstract";
  });

  // ---------- featured carousel ----------

  (function () {
    var track = $("[data-track]");
    var dots = $("[data-dots]");
    var cards = PAGE.featured.map(function (id, i) {
      var it = byId[id];
      var p = pair(it, i === 0 ? "auto" : "none");
      var card = el("figure", { class: "card" }, [p.node, caption(it, false)]);
      openOnActivate(p.node, it, function () {
        if (card.classList.contains("active")) return true;
        go(i);
        return false;
      });
      track.appendChild(card);
      var dot = el("button", { type: "button", "aria-label": "Show the " + it.species.toLowerCase() });
      dot.addEventListener("click", function () { go(i); });
      dots.appendChild(dot);
      return { card: card, video: p.video, dot: dot };
    });

    var current = 0, inView = false;
    function go(i) {
      i = (i + cards.length) % cards.length;
      var c = cards[i].card;
      track.scrollTo({ left: c.offsetLeft - (track.clientWidth - c.offsetWidth) / 2, behavior: reduceMotion ? "auto" : "smooth" });
    }
    function setActive(i) {
      current = i;
      cards.forEach(function (c, j) {
        var on = j === i;
        c.card.classList.toggle("active", on);
        c.dot.setAttribute("aria-current", on ? "true" : "false");
        if (on && inView) play(c.video); else pause(c.video);
      });
    }

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && e.intersectionRatio > 0.6) setActive(cards.findIndex(function (c) { return c.card === e.target; }));
        });
      }, { root: track, threshold: [0.6] });
      cards.forEach(function (c) { io.observe(c.card); });
      new IntersectionObserver(function (entries) { inView = entries[0].isIntersecting; setActive(current); }, { threshold: 0.2 }).observe(track);
    } else {
      inView = true;
    }

    $("[data-prev]").addEventListener("click", function () { go(current - 1); });
    $("[data-next]").addEventListener("click", function () { go(current + 1); });
    track.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { go(current + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { go(current - 1); e.preventDefault(); }
    });
    setActive(0);
  })();

  // ---------- frequency lens ----------

  (function () {
    var stage = $("[data-lens]");
    var base = $(".lens-base", stage);
    var top = $(".lens-top", stage);
    var picker = $("[data-lens-picker]");
    var layer = "rgb";
    var item = PAGE.lens[0];
    var touched = false;

    function src(it, kind) { return "static/lens/" + it.id + "_" + kind + ".jpg"; }
    function show() {
      base.src = src(item, layer);
      base.alt = item.species + " reference photograph, " + ({ rgb: "original", seg: "foreground only", hf: "high-frequency map" })[layer];
      top.src = src(item, "hf");
      stage.classList.toggle("layer-hf", layer === "hf");
    }

    PAGE.lens.forEach(function (it, i) {
      var b = el("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false" }, [
        el("img", { src: src(it, "rgb"), alt: "" }), document.createTextNode(it.species)
      ]);
      b.addEventListener("click", function () {
        item = it;
        $$("button", picker).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        show();
      });
      picker.appendChild(b);
      new Image().src = src(it, "hf");
    });

    $$("[data-layer]").forEach(function (b) {
      b.addEventListener("click", function () {
        layer = b.getAttribute("data-layer");
        $$("[data-layer]").forEach(function (x) { x.setAttribute("aria-checked", String(x === b)); });
        show();
      });
    });

    function moveTo(x, y) { stage.style.setProperty("--x", x + "px"); stage.style.setProperty("--y", y + "px"); }
    function fromEvent(e) {
      var r = stage.getBoundingClientRect();
      moveTo(Math.max(0, Math.min(r.width, e.clientX - r.left)), Math.max(0, Math.min(r.height, e.clientY - r.top)));
    }
    stage.addEventListener("pointermove", function (e) { touched = true; fromEvent(e); });
    stage.addEventListener("pointerdown", function (e) { touched = true; fromEvent(e); });

    // Until someone moves the lens, let it drift slowly over the animal.
    if (!reduceMotion) {
      var t0 = performance.now();
      (function orbit(now) {
        if (touched) return;
        var r = stage.getBoundingClientRect();
        var t = (now - t0) / 1000;
        moveTo(r.width * (0.5 + 0.2 * Math.cos(t * 0.6)), r.height * (0.48 + 0.14 * Math.sin(t * 0.9)));
        requestAnimationFrame(orbit);
      })(t0);
    }
    show();
  })();

  // ---------- gallery: autoplay in view, first half shown until expanded ----------

  (function () {
    var grid = $("[data-grid]");
    var chips = $("[data-chips]");
    var more = $("[data-more]");
    var tiles = PAGE.gallery.map(function (id) {
      var it = byId[id];
      var p = pair(it, "none", true);
      openOnActivate(p.node, it);
      var fig = el("figure", { class: "g-tile", "data-species": it.species }, [p.node, caption(it, true)]);
      grid.appendChild(fig);
      return { fig: fig, video: p.video };
    });

    var vids = tiles.map(function (t) { return t.video; }).filter(Boolean);
    autoplayInView(vids, 0.35);

    // Collapsed, the gallery shows the first half (an even number, so the two-column rows stay full).
    var limit = 2 * Math.ceil(tiles.length / 4);
    var filter = "All", expanded = false;

    function apply() {
      var matches = 0;
      tiles.forEach(function (t) {
        var match = filter === "All" || t.fig.getAttribute("data-species") === filter;
        if (match) matches++;
        t.fig.hidden = !(match && (expanded || matches <= limit));
        if (t.fig.hidden) pause(t.video);
      });
      var collapsible = matches > limit;
      more.hidden = !collapsible;
      grid.classList.toggle("collapsed", collapsible && !expanded);
      more.setAttribute("aria-expanded", String(expanded));
      more.textContent = "";
      more.appendChild(document.createTextNode(expanded ? "Show fewer" : "Show all " + matches + " examples"));
      more.appendChild(icon("down"));
    }

    more.addEventListener("click", function () {
      var before = more.getBoundingClientRect().top;
      expanded = !expanded;
      apply();
      // keep the button where it was, so collapsing does not throw the reader further down the page
      if (!expanded) window.scrollBy(0, more.getBoundingClientRect().top - before);
    });

    var species = ["All"].concat(Array.from(new Set(PAGE.gallery.map(function (id) { return byId[id].species; }))));
    species.forEach(function (s, i) {
      var b = el("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", text: s });
      b.addEventListener("click", function () {
        $$("button", chips).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        filter = s;
        apply();
      });
      chips.appendChild(b);
    });
    apply();
  })();

  // ---------- viewer ----------

  var viewer = $("#viewer");
  var vBody = $(".viewer-body", viewer);
  var vVideo = null;

  function openViewer(it) {
    vBody.textContent = "";
    var p = pair(it, "auto");
    p.node.removeAttribute("tabindex");
    p.node.removeAttribute("role");
    vVideo = p.video;
    if (vVideo) vVideo.controls = true;
    vBody.appendChild(p.node);
    vBody.appendChild(el("p", { class: "viewer-title", text: it.species }));
    vBody.appendChild(el("a", { class: "reference-id", href: it.reference, target: "_blank", rel: "noopener", text: "Original reference · " + (it.referenceId || it.id) }));
    $(".viewer-prompt", viewer).textContent = it.prompt;
    viewer.showModal();
    play(vVideo);
  }
  function closeViewer() {
    if (vVideo) {
      pause(vVideo);
      videos.splice(videos.indexOf(vVideo), 1);
      vVideo.removeAttribute("src");
      vVideo.load();
      vVideo = null;
    }
    viewer.close();
  }
  $(".viewer-close", viewer).addEventListener("click", closeViewer);
  viewer.addEventListener("click", function (e) { if (e.target === viewer) closeViewer(); });
  viewer.addEventListener("cancel", function (e) { e.preventDefault(); closeViewer(); });

  // ---------- reveal on scroll, count-up numbers ----------

  function countUp(node) {
    var target = +node.getAttribute("data-count");
    var t0 = performance.now(), dur = 1400;
    (function step(now) {
      var k = Math.min(1, (now - t0) / dur);
      node.textContent = Math.round(target * (1 - Math.pow(1 - k, 3))).toLocaleString("en-US");
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }

  if (!reduceMotion && "IntersectionObserver" in window) {
    var rio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add("in");
        $$("[data-count]", e.target).forEach(countUp);
        rio.unobserve(e.target);
      });
    }, { threshold: 0 });
    $$(".reveal").forEach(function (s) { rio.observe(s); });
  } else {
    $$(".reveal").forEach(function (s) { s.classList.add("in"); });
  }

  // ---------- copy BibTeX ----------

  $$("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = $(btn.getAttribute("data-copy")).textContent;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1500);
      });
    });
  });
})();
