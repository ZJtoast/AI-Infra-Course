"use strict";
const search = document.getElementById("find");
if (search) {
  const hits = document.getElementById("hits");
  search.addEventListener("input", () => {
    const terms = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    hits.replaceChildren();
    if (!terms.length) return;
    const found = [];
    for (const doc of window.courseIndex) {
      for (const [id, title, sample] of doc.items) {
        const label = title.toLocaleLowerCase();
        const body = [label, doc.title, sample].join(" ").toLocaleLowerCase();
        if (terms.every(term => body.includes(term))) {
          const score = terms.filter(term => label.includes(term)).length + (doc.n ? 0.25 : 0);
          found.push({doc, id, title, score});
        }
      }
    }
    found.sort((a, b) => b.score - a.score);
    if (!found.length) hits.textContent = "没有找到匹配的知识点，可尝试较短的术语。";
    for (const row of found.slice(0, 15)) {
      const link = document.createElement("a");
      const label = document.createElement("small");
      label.textContent = row.doc.n ? "第 " + row.doc.n + " 课" : "学习路线";
      link.href = row.doc.url + "#" + row.id;
      link.append(label, row.title);
      hits.append(link);
    }
  });
}
const draftFields = document.querySelectorAll("textarea[data-exercise]");
if (draftFields.length) {
  const prefix = "ai-infra-course:draft:v1:";
  const unsaved = new WeakSet();
  function showStatus(field, message) {
    const status = field.closest(".response").querySelector(".draft-status");
    if (status.textContent !== message) status.textContent = message;
  }
  function restoreDraft(field) {
    if (unsaved.has(field)) return;
    try {
      const draft = localStorage.getItem(prefix + field.dataset.exercise) || "";
      if (field.value !== draft) field.value = draft;
      showStatus(field, draft ? "已保存在本机" : "");
    } catch {
      showStatus(field, "仅本页暂存");
    }
  }
  function saveDraft(field) {
    const key = prefix + field.dataset.exercise;
    try {
      if (field.value) localStorage.setItem(key, field.value);
      else localStorage.removeItem(key);
      unsaved.delete(field);
      showStatus(field, field.value ? "已保存在本机" : "");
    } catch {
      unsaved.add(field);
      showStatus(field, "未保存，可复制留存");
    }
  }
  document.addEventListener("input", event => {
    if (event.target.matches("textarea[data-exercise]")) saveDraft(event.target);
  });
  window.addEventListener("pageshow", () => draftFields.forEach(restoreDraft));
  window.addEventListener("storage", event => {
    if (event.key === null) draftFields.forEach(restoreDraft);
    else if (event.key.startsWith(prefix)) {
      const field = document.getElementById("draft-" + event.key.slice(prefix.length));
      if (field) restoreDraft(field);
    }
  });
  // Print the complete worked solutions and restore the reading state afterwards.
  let printSolutions = [];
  window.addEventListener("beforeprint", () => {
    for (const item of document.querySelectorAll("details.solution:not([open])")) {
      printSolutions.push(item);
      item.open = true;
    }
  });
  window.addEventListener("afterprint", () => {
    for (const item of printSolutions) item.open = false;
    printSolutions = [];
  });
}
function initBookChapters() {
  const chapters = Array.from(document.querySelectorAll("article > .book-chapter"));
  if (!chapters.length) return;
  const menus = Array.from(document.querySelectorAll(".booknav .chap"));
  function displayChapter() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { id = ""; }
    const target = document.getElementById(id);
    const owner = target?.closest("[data-chapter]");
    const current = owner?.dataset.chapter || chapters[0].dataset.chapter;
    const switched = chapters.find(ch => ch.dataset.chapter === current)?.hidden;
    for (const ch of chapters) ch.hidden = ch.dataset.chapter !== current;
    for (const menu of menus) {
      menu.open = Boolean(owner) && menu.dataset.chapter === current;
      if (menu.open) menu.closest(".toc-module").open = true;
    }
    if (target && owner && (switched || target.closest("article"))) {
      target.scrollIntoView({block: "start", behavior: "instant"});
    }
  }
  displayChapter();
  window.addEventListener("hashchange", displayChapter);
}
initBookChapters();
function initReadingPosition(panel, article, header, options = {}) {
  // The book renders one chapter at a time. Only search the stops of that chapter.
  const chapters = new Map();
  const byAnchor = new Map();
  for (const link of panel.querySelectorAll('a[href^="#"]')) {
    const target = document.getElementById(link.hash.slice(1));
    if (!target || (!article.contains(target) && target.id !== "main")) continue;
    const chapter = target.closest(".book-chapter");
    if (!chapters.has(chapter)) chapters.set(chapter, []);
    const entry = {link, target};
    chapters.get(chapter).push(entry);
    byAnchor.set(target, entry);
  }
  let stops = [];
  let current;
  let requested;
  let frame = 0;

  function reveal() {
    if (!current || panel.hidden) return;
    const link = current.link;
    for (let group = link.closest("details"); group && panel.contains(group); group = group.parentElement.closest("details")) {
      group.open = true;
    }
    if (!link.getClientRects().length) return;
    const row = link.getBoundingClientRect();
    const bounds = panel.getBoundingClientRect();
    // Move only the table of contents. The document and keyboard focus stay in place.
    if (row.top < bounds.top + 6) panel.scrollTop += row.top - bounds.top - 6;
    else if (row.bottom > bounds.bottom - 6) panel.scrollTop += row.bottom - bounds.bottom + 6;
  }
  function select(entry) {
    if (!entry || entry === current) return;
    current?.link.removeAttribute("aria-current");
    current = entry;
    current.link.setAttribute("aria-current", "location");
    options.onSelect?.(current.link);
    reveal();
  }
  function update() {
    frame = 0;
    if (!stops.length) return;
    const scrollTop = window.scrollY;
    const lastScroll = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
    // A short final section can share its maximum scroll position with its neighbour.
    if (requested) {
      const {target} = requested;
      const margin = parseFloat(getComputedStyle(target).scrollMarginTop) || 0;
      const padding = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
      const top = target.getBoundingClientRect().top + scrollTop - padding - margin;
      if (Math.abs(scrollTop - Math.max(0, Math.min(lastScroll, top))) <= 3) {
        select(requested);
        return;
      }
      requested = null;
    }
    const line = (options.readingTop?.() ?? header.getBoundingClientRect().bottom) + 32;
    let lo = 0, hi = stops.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (stops[mid].target.getBoundingClientRect().top <= line) lo = mid + 1;
      else hi = mid;
    }
    const atEnd = lastScroll > 0 && scrollTop >= lastScroll - 2;
    select(stops[atEnd ? stops.length - 1 : Math.max(0, lo - 1)]);
  }
  function queue() {
    if (!frame) frame = requestAnimationFrame(update);
  }
  function navigation() {
    stops = chapters.get(article.querySelector(":scope > .book-chapter:not([hidden])")) || [];
    let target;
    try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch { /* Invalid fragment. */ }
    requested = byAnchor.get(target);
    if (requested) select(requested);
    queue();
  }
  window.addEventListener("scroll", queue, {passive: true});
  window.addEventListener("hashchange", navigation);
  window.addEventListener("pageshow", navigation);
  window.addEventListener("resize", queue);
  new ResizeObserver(queue).observe(article);
  // Activating a second link to the same fragment does not fire hashchange.
  panel.addEventListener("click", event => {
    const link = event.target.closest('a[href^="#"]');
    if (link && link.hash === location.hash) navigation();
  });
  navigation();
  return reveal;
}
function initReaderSidebar() {
  const layout = document.querySelector(".reader .layout");
  if (!layout) return;
  const panel = layout.querySelector("#reader-toc");
  const toggle = document.querySelector("[data-toc-toggle]");
  const divider = layout.querySelector(".toc-divider");
  const shade = layout.querySelector(".toc-shade");
  const article = layout.querySelector("article");
  const hero = document.querySelector(".articlehero");
  const header = document.querySelector(".top");
  const compact = matchMedia("(max-width: 700px)");
  const key = "ai-infra-course:toc:";
  function read(name) {
    try { return localStorage.getItem(key + name); } catch { return null; }
  }
  function remember(name, value) {
    try { localStorage.setItem(key + name, String(value)); } catch { /* Session controls still work. */ }
  }
  const widthKey = () => compact.matches ? "mobile-width" : "desktop-width";
  let preferred = Number(read(widthKey())) || 0;
  let shown = !compact.matches && read("desktop-open") !== "false";
  let limits = {min: 0, max: 0};
  let width = 0;
  let drag = null;
  let frame = 0;
  let drawerTarget = null;
  const revealReadingPosition = initReadingPosition(panel, article, header);

  function applyWidth(value) {
    width = Math.max(limits.min, Math.min(limits.max, Math.round(value)));
    layout.style.setProperty("--toc-width", width + "px");
    divider.setAttribute("aria-valuenow", String(width));
    divider.setAttribute("aria-valuetext", width + " 像素");
  }
  function measure() {
    const viewport = document.documentElement.clientWidth;
    const max = compact.matches
      ? Math.min(340, viewport - 36)
      : Math.min(340, layout.clientWidth * 0.38, layout.clientWidth - 360);
    limits.max = Math.max(0, Math.floor(max));
    limits.min = Math.min(180, limits.max);
    divider.setAttribute("aria-valuemin", String(limits.min));
    divider.setAttribute("aria-valuemax", String(limits.max));
    const top = Math.ceil(header.getBoundingClientRect().height);
    layout.style.setProperty("--toc-top", top + "px");
    document.documentElement.style.setProperty("--reader-scroll-top", top + 12 + "px");
    const defaultWidth = compact.matches ? Math.min(292, viewport * 0.80) : Math.max(212, Math.min(248, layout.clientWidth * 0.20));
    applyWidth(preferred || defaultWidth);
  }
  function display(open, persist = false) {
    shown = open;
    const overlay = open && compact.matches;
    panel.hidden = divider.hidden = !open;
    shade.hidden = !overlay;
    article.inert = overlay;
    if (hero) hero.inert = overlay;
    document.documentElement.classList.toggle("toc-overlay", overlay);
    layout.classList.toggle("toc-closed", !open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.textContent = open ? "收起目录" : "展开目录";
    if (persist && !compact.matches) remember("desktop-open", open);
    if (open) revealReadingPosition();
  }
  function closeDrawer() {
    display(false);
    toggle.focus({preventScroll: true});
  }
  function returnToIndex(target) {
    display(true);
    for (let d = target.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
    target.scrollIntoView({block: "nearest"});
    const entry = target.querySelector("a");
    if (entry) entry.focus({preventScroll: true});
  }
  function followFragment() {
    const target = document.getElementById(location.hash.slice(1));
    if (target && panel.contains(target)) returnToIndex(target);
    else if (target && article.contains(target) && compact.matches && shown) display(false);
    if (drawerTarget && target === drawerTarget) {
      drawerTarget.focus({preventScroll: true});
      drawerTarget = null;
    }
  }
  function finishDrag(event) {
    if (!drag || event.pointerId !== drag.id) return;
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    if (event.type === "pointerup") applyWidth(drag.startWidth + event.clientX - drag.x);
    drag = null;
    document.documentElement.classList.remove("toc-dragging");
    preferred = width;
    remember(widthKey(), width);
  }
  divider.addEventListener("pointerdown", event => {
    if (!event.isPrimary || event.button !== 0 || drag) return;
    divider.focus({preventScroll: true});
    drag = {id: event.pointerId, x: event.clientX, startWidth: width};
    divider.setPointerCapture(event.pointerId);
    document.documentElement.classList.add("toc-dragging");
    event.preventDefault();
  });
  divider.addEventListener("pointermove", event => {
    if (!drag || event.pointerId !== drag.id) return;
    drag.nextWidth = drag.startWidth + event.clientX - drag.x;
    if (!frame) frame = requestAnimationFrame(() => {
      frame = 0;
      if (drag) applyWidth(drag.nextWidth);
    });
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    divider.addEventListener(type, finishDrag);
  }
  divider.addEventListener("keydown", event => {
    const step = event.shiftKey ? 48 : 16;
    const keys = {ArrowLeft: width - step, ArrowRight: width + step, Home: limits.min, End: limits.max};
    if (!Object.hasOwn(keys, event.key)) return;
    event.preventDefault();
    applyWidth(keys[event.key]);
    preferred = width;
    remember(widthKey(), width);
  });

  layout.classList.add("toc-ready");
  toggle.hidden = false;
  display(shown);
  measure();
  toggle.addEventListener("click", () => display(!shown, true));
  const localHeading = panel.querySelector(".tocbox > summary");
  if (localHeading) localHeading.addEventListener("click", event => {
    event.preventDefault();
    display(false, true);
    toggle.focus({preventScroll: true});
  });
  shade.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && compact.matches && shown) closeDrawer();
  });
  layout.addEventListener("click", event => {
    const link = event.target.closest("a[href^='#']");
    if (!link || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    const target = document.getElementById(link.hash.slice(1));
    if (!target) return;
    if (panel.contains(target)) returnToIndex(target);
    else if (panel.contains(link) && compact.matches) {
      display(false);
      target.setAttribute("tabindex", "-1");
      if (target.closest("[hidden]")) drawerTarget = target;
      else target.focus({preventScroll: true});
    }
  });
  compact.addEventListener("change", () => {
    preferred = Number(read(widthKey())) || 0;
    display(!compact.matches && read("desktop-open") !== "false");
    measure();
  });
  window.addEventListener("resize", measure);
  new ResizeObserver(measure).observe(header);
  window.addEventListener("hashchange", followFragment);
  followFragment();
}
initReaderSidebar();

function initHomeCatalog() {
  const panel = document.getElementById("catalog-toc");
  if (!panel) return;
  const groups = document.querySelector(".catalog-groups");
  const menu = panel.closest("details");
  const summary = menu.querySelector("summary");
  const label = summary.querySelector("[data-current-module]");
  const header = document.querySelector(".top");
  const compact = matchMedia("(max-width: 700px)");
  function measure() {
    const top = Math.ceil(header.getBoundingClientRect().height) + 8;
    document.documentElement.style.setProperty("--catalog-top", top + "px");
    const space = compact.matches ? summary.offsetHeight + 8 : 0;
    document.documentElement.style.setProperty("--catalog-anchor", top + space + "px");
  }
  function selectLayout() {
    menu.open = !compact.matches;
    measure();
  }
  selectLayout();
  compact.addEventListener("change", selectLayout);
  new ResizeObserver(measure).observe(header);
  new ResizeObserver(measure).observe(summary);
  initReadingPosition(panel, groups, header, {
    onSelect(link) { label.textContent = link.textContent; },
    readingTop() { return compact.matches ? Math.max(header.getBoundingClientRect().bottom, summary.getBoundingClientRect().bottom) : header.getBoundingClientRect().bottom; }
  });
  panel.addEventListener("click", event => {
    const link = event.target.closest('a[href^="#"]');
    if (link && compact.matches && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
      menu.open = false;
      document.getElementById(link.hash.slice(1))?.focus({preventScroll: true});
    }
  });
  menu.addEventListener("keydown", event => {
    if (event.key === "Escape" && menu.open) {
      menu.open = false;
      summary.focus({preventScroll: true});
    }
  });
}
initHomeCatalog();

document.addEventListener("click", async e => {
  const toggle = e.target.closest("[data-toggle-notes]");
  if (toggle) {
    const enabled = toggle.closest(".code").classList.toggle("show-reading-notes");
    toggle.setAttribute("aria-pressed", String(enabled));
    return;
  }
  const button = e.target.closest("[data-copy]");
  if (!button) return;
  const code = button.closest(".code").querySelector("code");
  const value = Array.from(code.querySelectorAll(".cl"), line => line.textContent).join("\n");
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(value);
    else {
      const node = document.createElement("textarea");
      node.value = value; node.style.position = "fixed"; node.style.opacity = "0";
      document.body.append(node); node.select(); const ok = document.execCommand("copy"); node.remove();
      if (!ok) throw Error("copy");
    }
    button.textContent = "已复制";
  } catch {
    const selection = getSelection(), range = document.createRange();
    range.selectNodeContents(code); selection.removeAllRanges(); selection.addRange(range);
    button.textContent = "已选中，请复制";
  }
  setTimeout(() => { button.textContent = "复制代码"; }, 2200);
});


(() => {
  const page = document.querySelector("article");
  if (!page) return;
  const card = document.createElement("aside");
  card.className = "term-card";
  card.id = "active-term-card";
  card.setAttribute("role","dialog");
  card.setAttribute("aria-labelledby","term-name");
  card.setAttribute("aria-describedby","term-meaning");
  card.hidden = true;
  card.innerHTML = '<div class="term-top"><strong id="term-name"></strong><button type="button" aria-label="关闭解释">×</button></div><div id="term-meaning"></div>';
  document.body.append(card);
  const name = card.querySelector("strong");
  const meaning = card.querySelector("#term-meaning");
  const symbolLists = new WeakMap();
  const close = card.querySelector("button");
  let current, frame;
  function hide(restore = false) {
    if (!current) return;
    const previous = current;
    current = null;
    previous.setAttribute("aria-expanded","false");
    previous.removeAttribute("aria-controls");
    card.hidden = true;
    if (restore) previous.focus({preventScroll:true});
  }
  function locate() {
    if (!current) return;
    const r = current.getBoundingClientRect(), c = card.getBoundingClientRect();
    const viewportY = window.visualViewport?.offsetTop || 0;
    const bottom = viewportY + (window.visualViewport?.height || innerHeight);
    if (r.bottom < viewportY || r.top > bottom) { hide(); return; }
    card.style.left = Math.max(12,Math.min(r.left,document.documentElement.clientWidth-c.width-12)) + "px";
    const below = r.bottom+9;
    card.style.top = (below+c.height <= bottom-12 ? below : Math.max(viewportY+10,r.top-c.height-9)) + "px";
  }
  function queue() { if (current) { cancelAnimationFrame(frame); frame=requestAnimationFrame(locate); } }
  function open(link) {
    if (link === current) { hide(true); return; }
    hide(); current = link;
    card.classList.toggle("symbol-card",link.hasAttribute("data-symbols"));
    if (card.classList.contains("symbol-card")) {
      const section=link.closest("section.topic") || link.closest("section");
      if (!symbolLists.has(section)) {
        symbolLists.set(section, JSON.parse(section.querySelector("template.symbol-notes").content.textContent));
      }
      const notes=symbolLists.get(section);
      name.textContent="符号说明";
      const list=document.createElement("dl");
      for (const index of link.dataset.symbols.split(" ")) {
        const [label,body,format,labelMath]=notes[Number(index)];
        const term=document.createElement("dt"), definition=document.createElement("dd");
        if (labelMath) term.innerHTML=labelMath;
        else term.textContent=label;
        if (format === "mathml") definition.innerHTML=body;
        else definition.textContent=body;
        list.append(term,definition);
      }
      meaning.replaceChildren(list);
    } else {
      name.textContent=link.textContent;
      const def=document.getElementById(link.hash.slice(1));
      if (!def || !def.nextElementSibling) { hide(); return; }
      meaning.textContent=def.nextElementSibling.textContent;
    }
    card.hidden=false;
    card.scrollTop=0;
    current.setAttribute("aria-expanded","true");
    current.setAttribute("aria-controls",card.id);
    locate();
    close.focus({preventScroll:true});
  }
  for (const a of page.querySelectorAll("a.term")) { a.setAttribute("aria-haspopup","dialog"); a.setAttribute("aria-expanded","false"); }
  document.addEventListener("click", e=>{
    const a=e.target.closest("a.term,[data-symbols]");
    if (a && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey && e.button===0) {e.preventDefault();open(a);}
    else if (!card.contains(e.target)) hide();
  });
  close.addEventListener("click",()=>hide(true));
  document.addEventListener("keydown",e=>{
    if ((e.key==="Enter" || e.key===" ") && e.target.matches("[data-symbols]")) {
      e.preventDefault(); open(e.target); return;
    }
    if (!current) return;
    if(e.key==="Escape"){e.preventDefault();hide(true);}
    if(e.key==="Tab" && e.target===close){const trigger=current; hide(); trigger.focus({preventScroll:true});}
  });
  window.addEventListener("resize",queue);
  document.addEventListener("scroll",queue,true);
  window.visualViewport?.addEventListener("resize",queue);
})();

// A source range directly beside its explanation needs only a line label.
function initSourceDistances() {
  const refs = document.querySelectorAll("a.line-ref[data-source-figures]");
  if (!refs.length) return;
  const targets = new WeakMap();
  const visible = new Set();
  let frame = 0;
  function refresh() {
    frame = 0;
    const changes = [];
    for (const ref of visible) {
      if (ref.getAttribute("aria-expanded") === "true") continue;
      const box = ref.getBoundingClientRect();
      if (!box.height) continue;
      const lineHeight = parseFloat(getComputedStyle(ref).lineHeight);
      const near = targets.get(ref).some(({first, last, pre}) => {
        const a = first.getBoundingClientRect(), b = last.getBoundingClientRect();
        const clip = pre.getBoundingClientRect();
        if (!a.height || a.top < clip.top - 1 || b.bottom > clip.bottom + 1) return false;
        const gap = Math.max(0, a.top - box.bottom, box.top - b.bottom);
        const span = Math.max(b.bottom, box.bottom) - Math.min(a.top, box.top);
        return gap <= lineHeight * 2.5 && span <= window.innerHeight * 0.8;
      });
      if (near !== ref.classList.contains("line-location")) changes.push([ref, near]);
    }
    for (const [ref, near] of changes) {
      ref.classList.toggle("line-location", near);
      ref.classList.toggle("line-ref", !near);
      if (near) {
        ref.dataset.sourceHref = ref.getAttribute("href");
        ref.removeAttribute("href");
        ref.removeAttribute("aria-expanded");
      } else {
        ref.setAttribute("href", ref.dataset.sourceHref);
        ref.setAttribute("aria-expanded", "false");
      }
    }
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(refresh);
  }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) visible.add(entry.target);
      else visible.delete(entry.target);
    }
    schedule();
  }, {rootMargin: "600px 0px"});
  for (const ref of refs) {
    const ranges = [];
    for (const id of ref.dataset.sourceFigures.split(" ")) {
      const figure = document.getElementById(id);
      const first = figure?.querySelector('.cl[data-line="' + ref.dataset.firstLine + '"]');
      const last = figure?.querySelector('.cl[data-line="' + ref.dataset.lastLine + '"]');
      if (first && last) ranges.push({first, last, pre: first.closest("pre")});
    }
    targets.set(ref, ranges);
    observer.observe(ref);
  }
  new ResizeObserver(schedule).observe(document.querySelector("article"));
  window.addEventListener("resize", schedule);
  document.addEventListener("toggle", schedule, true);
  document.addEventListener("click", schedule);
  document.addEventListener("scroll", event => {
    if (event.target.tagName === "PRE") schedule();
  }, true);
  document.fonts.ready.then(schedule);
}
initSourceDistances();

// Keep the explanation and the passage it refers to in the same reading position.
(() => {
  let trigger, panel;
  function dismiss(restoreFocus = false) {
    if (!trigger) return;
    (panel.closest('tr.reading-detail-row') || panel).remove();
    trigger.setAttribute('aria-expanded', 'false');
    trigger.removeAttribute('aria-controls');
    if (restoreFocus) trigger.focus({preventScroll: true});
    trigger = panel = null;
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('a.line-ref, a.example-ref');
    if (!link || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (link === trigger) { dismiss(true); return; }
    dismiss();
    trigger = link;
    panel = document.createElement('aside');
    panel.className = 'reading-detail';
    panel.id = 'active-reading-detail';
    const codeRef = link.classList.contains('line-ref');
    const label = codeRef ? [link.dataset.languageLabel, link.dataset.file, link.textContent].filter(Boolean).join(' · ') : link.textContent;
    panel.setAttribute('aria-label', label);
    const bar = document.createElement('header');
    const title = document.createElement('strong');
    title.textContent = label;
    const close = document.createElement('button');
    close.type = 'button';
    close.textContent = '收起';
    close.addEventListener('click', () => dismiss(true));
    bar.append(title, close);
    panel.append(bar);
    if (codeRef) {
      const block = document.createElement('pre');
      const code = document.createElement('code');
      const rows = JSON.parse(link.dataset.rows);
      const syntax = link.dataset.syntax ? JSON.parse(link.dataset.syntax) : rows.map(([, source]) => [[null, source]]);
      if (link.dataset.language) code.className = 'language-' + link.dataset.language;
      const commonIndent = Math.min(...rows.filter(([, source]) => source.trim()).map(([, source]) => source.match(/^\s*/)[0].length));
      for (const [index, [number, , note]] of rows.entries()) {
        const row = document.createElement('span');
        row.className = 'reading-line';
        const gutter = document.createElement('span');
        gutter.className = 'reading-number';
        gutter.textContent = number;
        const annotation = document.createElement('span');
        annotation.className = 'reading-note';
        annotation.textContent = '  ' + link.dataset.comment + ' ' + note;
        const statement = document.createElement('span');
        statement.className = 'reading-statement';
        let skip = commonIndent;
        for (const [role, text] of syntax[index]) {
          const fragment = text.slice(skip);
          skip = Math.max(0, skip - text.length);
          if (!fragment) continue;
          if (role) {
            const token = document.createElement('span');
            token.className = 'tok-' + role;
            token.textContent = fragment;
            statement.append(token);
          } else {
            statement.append(document.createTextNode(fragment));
          }
        }
        row.append(gutter, statement, annotation);
        code.append(row);
      }
      block.append(code);
      panel.append(block);
    } else {
      const example = document.createElement('p');
      example.textContent = link.dataset.example;
      panel.append(example);
    }
    const row = link.closest('tr');
    if (row) {
      const expansion = document.createElement('tr');
      expansion.className = 'reading-detail-row';
      const cell = document.createElement('td');
      cell.colSpan = Array.from(row.cells).reduce((count, column) => count + column.colSpan, 0);
      cell.append(panel);
      expansion.append(cell);
      row.before(expansion);
    } else {
      (link.closest('p, li, figcaption') || link).before(panel);
    }
    trigger.setAttribute('aria-expanded', 'true');
    trigger.setAttribute('aria-controls', panel.id);
    close.focus({preventScroll: true});
    if (panel.getBoundingClientRect().top < 70 || panel.getBoundingClientRect().top > innerHeight - 100) {
      panel.scrollIntoView({block: 'start'});
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && trigger) { event.preventDefault(); dismiss(true); }
  });
})();

// Coordinates below use the 900 × 390 drawing surface of the two teaching figures.
for (const figure of document.querySelectorAll('figure[data-trace]')) {
  const link = figure.querySelector('a');
  const image = link.querySelector('img');
  const surface = document.createElement('span');
  surface.className = 'trace-surface';
  image.replaceWith(surface);
  surface.append(image);
  const ns = 'http://www.w3.org/2000/svg';
  const overlay = document.createElementNS(ns, 'svg');
  overlay.setAttribute('viewBox', '0 0 900 390');
  overlay.setAttribute('aria-hidden', 'true');
  surface.append(overlay);
  const controls = document.createElement('div');
  controls.className = 'trace-controls';
  const result = document.createElement('output');
  function slider(label, max, initial) {
    const field = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = max;
    input.value = initial;
    field.append(label, input);
    controls.append(field);
    return input;
  }
  function marker(tag, geometry) {
    const shape = document.createElementNS(ns, tag);
    for (const [attr, value] of Object.entries(geometry)) shape.setAttribute(attr, value);
    overlay.append(shape);
  }
  let render;
  if (figure.dataset.trace === 'rope-frequency-pairs') {
    const position = slider('位置 p', 12, 3);
    render = () => {
      const p = Number(position.value);
      overlay.replaceChildren();
      for (const [i, rate] of [1, 0.01].entries()) {
        marker('circle', {cx: 75 + 432*i + 337*p/12, cy: 223 - 59*Math.cos(p*rate), r: 7});
      }
      result.textContent = '位置 ' + p + '：两对通道的转角分别为 ' + p + ' 和 ' + (p/100).toFixed(2) +
        ' 弧度，对应的余弦为 ' + Math.cos(p).toFixed(4) + ' 和 ' + Math.cos(p/100).toFixed(4) + '。';
    };
  } else if (figure.dataset.trace === 'ulysses-position-heads') {
    const position = slider('位置 p', 7, 6);
    const head = slider('头 H', 3, 0);
    render = () => {
      const p = Number(position.value), h = Number(head.value);
      const src = Math.floor(p/4), dst = Math.floor(h/2);
      overlay.replaceChildren();
      for (let i = 0; i < 2; i++) marker('rect', {x: 90 + 437*i + 68*h, y: 163 + 19*p, width: 68, height: 19});
      result.textContent = '位置 ' + p + '、头 ' + h + (src === dst ? ' 在交换前后都归 rank ' + src + '。' :
        ' 从 rank ' + src + ' 发往 rank ' + dst + '。') + '同一对坐标在两幅图中均已圈出。';
    };
  } else continue;
  controls.append(result);
  link.after(controls);
  controls.addEventListener('input', render);
  render();
}

document.querySelectorAll('.state-deck').forEach(deck=>{
  const pages=Array.from(deck.children).filter(el=>el.tagName==='DETAILS');
  if(!pages.length)return;
  const tabs=document.createElement('div');
  tabs.className='state-tabs'; tabs.setAttribute('role','tablist');
  tabs.setAttribute('aria-label',deck.querySelector('h4').textContent);
  const buttons=pages.map((page,i)=>{
    const button=document.createElement('button');
    button.type='button';button.id=page.id+'-tab';
    button.setAttribute('role','tab');button.setAttribute('aria-controls',page.id);
    button.textContent=page.querySelector('summary').textContent;
    button.addEventListener('click',()=>pick(i,false));
    button.addEventListener('keydown',event=>{
      const n=pages.length;
      const next={ArrowLeft:(i+n-1)%n,ArrowRight:(i+1)%n,Home:0,End:n-1}[event.key];
      if(next!==undefined){event.preventDefault();pick(next,true);}
    });
    tabs.append(button);return button;
  });
  function pick(index,focus){
    pages.forEach((page,i)=>{
      page.open=i===index;
      page.hidden=i!==index;
      buttons[i].tabIndex=i===index?0:-1;
      buttons[i].setAttribute('aria-selected',String(i===index));
    });
    if(focus)buttons[index].focus();
  }
  for(const page of pages){page.removeAttribute('name');page.setAttribute('role','tabpanel');page.setAttribute('aria-labelledby',page.id+'-tab');}
  deck.insertBefore(tabs,pages[0]);deck.dataset.tabs='';pick(0,false);
});

/* Two query rows by two keys per tile. Keep the display and the recurrence on the same operands. */
(() => {
 const fmt=n=>n===-Infinity?'−∞':Number(n.toFixed(2)).toString().replace('-','−');
 const div=(cls,txt)=>{const el=document.createElement('div');el.className=cls;if(txt!==undefined)el.textContent=txt;return el;};
 function matrix(title,values,rows,cols,style){
   const figure=div('mx-matrix');figure.append(div('mx-label',title));
   const table=document.createElement('table');table.setAttribute('aria-label',title);
   const header=table.createTHead().insertRow();
   header.append(document.createElement('td'));
   function head(parent,name,scope){const th=document.createElement('th');th.scope=scope;th.textContent=name;parent.append(th);}
   cols.forEach(c=>head(header,c,'col'));const body=table.createTBody();
   values.forEach((vals,r)=>{const tr=body.insertRow();head(tr,rows[r],'row');vals.forEach((v,c)=>{
     const cell=tr.insertCell();cell.className=style?.(r,c)||'';cell.textContent=typeof v==='number'?fmt(v):v;
   });});figure.append(table);return figure;
 }
 for(const lab of document.querySelectorAll('[data-matrix-lab]')){
   const [Q,K,V]=['q','k','v'].map(key=>JSON.parse(lab.dataset[key]));
   const S=Q.map(q=>K.map(k=>(q[0]*k[0]+q[1]*k[1])/Math.sqrt(2)));
   function rowStep(row,block,causal,old){
     const keys=[2*block,2*block+1];
     const scores=keys.map(j=>causal&&j>row?-Infinity:S[row][j]);
     const m=Math.max(old.m,...scores);
     const alpha=old.z?Math.exp(old.m-m):0;
     const weights=scores.map(s=>s===-Infinity?0:Math.exp(s-m));
     const local=[0,1].map(d=>weights.reduce((u,w,t)=>u+w*V[keys[t]][d],0));
     return {m,alpha,weights,local,z:alpha*old.z+weights[0]+weights[1],u:local.map((v,d)=>v+alpha*old.u[d])};
   }
   const pick=lab.querySelectorAll('button[data-tile]'),mask=lab.querySelector('input'),slot=lab.querySelector('.mx-output');
   let qr=1,kb=1,selected=null;
   function render(){
     const ii=[qr*2,qr*2+1],jj=[kb*2,kb*2+1],causal=mask.checked;
     const init=()=>({m:-Infinity,z:0,u:[0,0]});
     const old=ii.map(i=>kb?rowStep(i,0,causal,init()):init());
     const cur=ii.map((i,r)=>rowStep(i,kb,causal,old[r]));
     for(const b of pick){const on=b.dataset.tile===String(2*qr+kb);b.setAttribute('aria-pressed',String(on));}
     const coord=x=>x.map(String),all=[0,1,2,3],d=['d₀','d₁'],out=['v₀','v₁'];
     const inside=(r,c)=>ii.includes(r)&&jj.includes(c);
     const section=(title,...children)=>{const e=div('mx-section');e.append(div('mx-subtitle',title));
         const operands=div('mx-operands');operands.append(...children);e.append(operands);return e;};
     const symbol=s=>div('mx-symbol',s);
     const sview=S.map((row,r)=>row.map((v,c)=>causal&&c>r?'屏蔽':(!inside(r,c)&&!(ii.includes(r)&&c<2*kb)?'·':v)));
     const inputs=section('从全矩阵取出这两行 Q 和两行 K',
       matrix('Q · query × 通道',Q,coord(all),d,(r,c)=>ii.includes(r)?'mx-q':''),
       symbol('×'),matrix('Kᵀ · 通道 × key',[K.map(k=>k[0]),K.map(k=>k[1])],d,coord(all),(r,c)=>jj.includes(c)?'mx-k':''),
       symbol('→'),matrix('S · 缩放点积',sview,coord(all),coord(all),(r,c)=>causal&&c>r?'mx-mask':inside(r,c)?'mx-now':ii.includes(r)&&c<2*kb?'mx-old':''));
     const mats=inputs.querySelectorAll('table');
     for(let r=0;r<4;r++)for(let c=0;c<4;c++){
       const cell=mats[2].tBodies[0].rows[r].cells[c+1],b=document.createElement('button');
       const value=cell.textContent;b.type='button';b.className='mx-point';b.textContent=value;b.dataset.coordinate=r+','+c;
       b.setAttribute('aria-label','追踪 query '+r+' 与 key '+c);
       const on=selected&&selected[0]===r&&selected[1]===c;b.setAttribute('aria-pressed',String(Boolean(on)));
       b.addEventListener('click',()=>{
         qr=Math.floor(r/2);kb=Math.floor(c/2);selected=[r,c];render();
         slot.querySelector('[data-coordinate="'+r+','+c+'"]').focus({preventScroll:true});
       });cell.replaceChildren(b);
     }
     let trace;
     if(selected){
       const [r,c]=selected;
       for(let h=0;h<2;h++){
         mats[0].tBodies[0].rows[r].cells[h+1].classList.add('mx-outline');
         mats[1].tBodies[0].rows[h].cells[c+1].classList.add('mx-outline');
       }
       trace=div('mx-trace',causal&&c>r?
        '追踪格 ('+r+', '+c+')：key 在这条 query 的未来，本格被屏蔽，V 的第 '+c+' 行对这个输出没有贡献。':
        '追踪格 ('+r+', '+c+')：固定 Q 的第 '+r+' 行和 Kᵀ 的第 '+c+' 列。分数为 ('+Q[r][0]+' × '+K[c][0]+' + '+Q[r][1]+' × '+K[c][1]+') / √2 = '+fmt(S[r][c])+'。此格的权重随后乘 V 的第 '+c+' 行。');
     }else trace=div('mx-trace','点选 S 中任意一格，可追踪它对应的 Q 行、Kᵀ 列和 V 行。');
     const pv=section('权重沿 key 列，和 V 的相同行号相乘',
       matrix('E · 当前基准下的指数权重',cur.map(v=>v.weights),coord(ii),coord(jj),(r,c)=>causal&&jj[c]>ii[r]?'mx-mask':'mx-now'),
       symbol('×'),matrix('V · 本块的两行',jj.map(j=>V[j]),coord(jj),out,()=> 'mx-k'),
       symbol('='),matrix('E V · 本块分子贡献',cur.map(v=>v.local),coord(ii),out,()=> 'mx-q'));
     if(selected){
       const [r,c]=selected,t=pv.querySelectorAll('table');
       t[0].tBodies[0].rows[r%2].cells[1+c%2].classList.add('mx-outline');
       for(let d=0;d<2;d++)t[1].tBodies[0].rows[c%2].cells[1+d].classList.add('mx-outline');
     }
     const state=section('换到同一基准后，累加这两行',
       matrix('旧状态',old.map(o=>[o.m,o.z,...o.u]),coord(ii),['m','z','U₀','U₁'],()=> 'mx-old'),
       symbol('→'),matrix('旧项倍率',cur.map(o=>[o.alpha]),coord(ii),['α'],()=> 'mx-now'),
       symbol('→'),matrix('合并后',cur.map(o=>[o.m,o.z,...o.u]),coord(ii),['m','z','U₀','U₁'],()=> 'mx-q'));
     const last=section(kb?'该行块已遍历全部 key，写回 O':'只处理了第一列块，目前为部分输出',
       matrix(kb?'最终输出 O':'当前 U / z',cur.map(s=>s.u.map(v=>s.z?v/s.z:'未定义')),coord(ii),out,()=> 'mx-q'));
     const note=div('mx-explain',cur.every(s=>s.weights.every(v=>v===0))?'这个分数块全部落在因果边界外，E 为零，已有分母和分子原样传到下一步。':
       '取每行新最大值 m 后，旧分母与旧分子同乘 α。加上 E 的行和得到新分母，加上矩阵积 E V 得到新分子。');
     slot.replaceChildren(inputs,trace,pv,state,note,last);
     lab.querySelector('.mx-live').textContent='正在查看 query '+ii.join('、')+'，key '+jj.join('、')+(causal?'，因果掩码已开启':'，全部位置可见');
   }
   pick.forEach(b=>b.addEventListener('click',()=>{const tile=Number(b.dataset.tile);qr=Math.floor(tile/2);kb=tile%2;selected=null;render();}));
   mask.addEventListener('change',render);render();
   lab.querySelector('.mx-controls').hidden=false;
 }
})();
