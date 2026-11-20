(() => {
  const storageKey = "ai-infra-course:appearance";
  const root = document.documentElement;
  const accents = ["blue", "teal", "green", "purple", "rose", "orange"];

  function apply(saved) {
    let value;
    try { value = JSON.parse(saved); } catch { /* Use the defaults. */ }
    root.dataset.mode = value?.mode === "dark" ? "dark" : "light";
    root.dataset.accent = accents.includes(value?.accent) ? value.accent : "blue";
    document.querySelectorAll(".appearance input").forEach(input => {
      input.checked = input.value === root.dataset[input.name];
    });
  }

  let saved;
  try { saved = localStorage.getItem(storageKey); } catch { /* Storage is optional. */ }
  apply(saved);

  document.addEventListener("DOMContentLoaded", () => {
    const picker = document.querySelector(".appearance");
    const trigger = picker.querySelector("summary");
    apply(JSON.stringify({mode: root.dataset.mode, accent: root.dataset.accent}));
    picker.addEventListener("change", event => {
      const input = event.target;
      if (!input.matches("input[type=radio]")) return;
      root.dataset[input.name] = input.value;
      const {mode, accent} = root.dataset;
      try { localStorage.setItem(storageKey, JSON.stringify({mode, accent})); } catch { /* Keep this page's selection. */ }
    });
    document.addEventListener("click", event => {
      if (picker.open && !picker.contains(event.target)) picker.open = false;
    });
    picker.addEventListener("focusout", event => {
      if (event.relatedTarget && !picker.contains(event.relatedTarget)) picker.open = false;
    });
    document.addEventListener("keydown", event => {
      if (picker.open && event.key === "Escape") {
        picker.open = false;
        trigger.focus();
        event.stopImmediatePropagation();
      }
    }, true);
  });
  window.addEventListener("storage", event => {
    if (event.key === storageKey || event.key === null) apply(event.newValue);
  });
})();
