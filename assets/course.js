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

  function applyWidth(value) {
    width = Math.max(limits.min, Math.min(limits.max, Math.round(value)));
    layout.style.setProperty("--toc-width", width + "px");
    divider.setAttribute("aria-valuenow", String(width));
    divider.setAttribute("aria-valuetext", width + " 像素");
  }
  function measure() {
    const viewport = document.documentElement.clientWidth;
    const max = compact.matches
      ? Math.min(520, viewport - 36)
      : Math.min(560, layout.clientWidth * 0.52, layout.clientWidth - 360);
    limits.max = Math.max(0, Math.floor(max));
    limits.min = Math.min(180, limits.max);
    divider.setAttribute("aria-valuemin", String(limits.min));
    divider.setAttribute("aria-valuemax", String(limits.max));
    const top = Math.ceil(header.getBoundingClientRect().height);
    layout.style.setProperty("--toc-top", top + "px");
    document.documentElement.style.setProperty("--reader-scroll-top", top + 12 + "px");
    applyWidth(preferred || (compact.matches ? viewport * 0.84 : layout.clientWidth * 0.26));
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
      target.focus({preventScroll: true});
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

document.addEventListener("click", async e => {
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
  card.innerHTML = '<div class="term-top"><strong id="term-name"></strong><button type="button" aria-label="关闭术语释义">×</button></div><p id="term-meaning"></p>';
  document.body.append(card);
  const name = card.querySelector("strong");
  const meaning = card.querySelector("p");
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
    name.textContent=link.textContent;
    const def=document.getElementById(link.hash.slice(1));
    if (!def || !def.nextElementSibling) { hide(); return; }
    meaning.textContent=def.nextElementSibling.textContent;
    card.hidden=false;
    current.setAttribute("aria-expanded","true");
    current.setAttribute("aria-controls",card.id);
    locate();
    close.focus({preventScroll:true});
  }
  for (const a of page.querySelectorAll("a.term")) { a.setAttribute("aria-haspopup","dialog"); a.setAttribute("aria-expanded","false"); }
  document.addEventListener("click", e=>{
    const a=e.target.closest("a.term");
    if (a && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey && e.button===0) {e.preventDefault();open(a);}
    else if (!card.contains(e.target)) hide();
  });
  close.addEventListener("click",()=>hide(true));
  document.addEventListener("keydown",e=>{
    if (!current) return;
    if(e.key==="Escape"){e.preventDefault();hide(true);}
    if(e.key==="Tab" && e.target===close){const trigger=current; hide(); trigger.focus({preventScroll:true});}
  });
  window.addEventListener("resize",queue);
  document.addEventListener("scroll",queue,true);
  window.visualViewport?.addEventListener("resize",queue);
})();

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
