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
function revealTarget() {
  const target = document.getElementById(location.hash.slice(1));
  if (!target) return;
  const parent = target.closest("details");
  if (parent && !parent.open) {
    parent.open = true;
    target.scrollIntoView({block:"nearest"});
  }
}
const localToc = document.querySelector(".tocbox");
if (localToc && matchMedia("(max-width:700px)").matches) localToc.open = false;
window.addEventListener("hashchange", revealTarget);
revealTarget();
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

