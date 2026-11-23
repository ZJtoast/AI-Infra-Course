(() => {
  "use strict";

  const ns = "http://www.w3.org/2000/svg";
  const openingDelay = 1300;
  const showerDuration = 5900;
  const clearingDelay = 450;
  let shower = null;

  function svgNode(tag, attributes) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }

  function artwork(reduced) {
    const svg = svgNode("svg", {
      viewBox: "0 0 108 112", preserveAspectRatio: "none",
      class: "aira-rain", "data-phase": "raining", "aria-hidden": "true", focusable: "false",
    });
    Object.assign(svg.style, {
      position: "fixed", zIndex: "7", pointerEvents: "none", overflow: "hidden",
      visibility: "hidden", userSelect: "none",
    });
    const drops = svgNode("g", {"data-rain-drops": "", "stroke-linecap": "round"});
    const strokes = [
      [24, 8, 41, .31], [40, 16, 57, .69], [52, 5, 61, .12], [69, 18, 54, .47],
      [79, 10, 43, .76], [32, 7, 65, .92], [58, 16, 57, .81], [73, 5, 59, .21],
      [46, 12, 48, .38], [88, 7, 45, .59], [21, 6, 40, .64], [61, 8, 52, .03],
    ];
    const animations = [];
    for (const [x, length, fall, phase] of strokes) {
      const rail = svgNode("g", {});
      rail.append(svgNode("path", {
        d: `M ${x} 29 l -2.7 ${length}`,
        stroke: "#4588ae", "stroke-width": "1.8", opacity: ".7",
      }));
      rail.append(svgNode("path", {
        d: `M ${x + .34} 29 l -2.7 ${length}`,
        stroke: "#c6e8f4", "stroke-width": ".8", opacity: ".95",
      }));
      if (reduced) {
        rail.setAttribute("transform", `translate(${-phase * 9} ${phase * fall})`);
      } else {
        animations.push(() => rail.animate([
          { transform: "translate(0px,0px)", opacity: .1, offset: 0 },
          { opacity: .92, offset: .16 },
          { opacity: .82, offset: .78 },
          { transform: `translate(-9px,${fall}px)`, opacity: 0, offset: 1 },
        ], {
          duration: 730 + length * 11, delay: -phase * (730 + length * 11),
          iterations: Infinity, easing: "linear",
        }));
      }
      drops.append(rail);
    }
    const cloud = svgNode("g", {"data-rain-cloud": ""});
    cloud.append(svgNode("path", {
      d: "M25 32C19 32 16 29 16 25C16 20 20 16 25 17C25 11 30 7 36 8C40 2 50 1 56 7C62 3 73 6 76 14C84 12 91 18 91 24C91 29 87 32 80 32Z",
      fill: "#c5d5e1", "fill-opacity": ".93", stroke: "#668299", "stroke-width": "1.3",
      "stroke-linejoin": "round",
    }));
    cloud.append(svgNode("path", {
      d: "M21 25Q22 19 28 21Q28 12 37 12Q44 5 53 10",
      fill: "none", stroke: "#eef5f9", "stroke-width": "2.3", "stroke-linecap": "round",
    }));
    svg.append(drops, cloud);
    return {svg, drops, cloud, animations};
  }

  function cancel() {
    if (!shower) return;
    const scene = shower;
    shower = null;
    for (const timer of scene.timers) clearTimeout(timer);
    for (const motion of scene.motions) motion.cancel();
    scene.observer?.disconnect();
    window.removeEventListener("resize", scene.position);
    document.removeEventListener("scroll", scene.position, true);
    window.visualViewport?.removeEventListener("resize", scene.position);
    window.visualViewport?.removeEventListener("scroll", scene.position);
    document.removeEventListener("visibilitychange", scene.checkVisibility);
    window.removeEventListener("pagehide", cancel);
    scene.svg.remove();
  }

  function start({anchor, onOpen, onStop}) {
    cancel();
    if (!anchor?.isConnected || document.hidden || !anchor.getClientRects().length) return false;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const {svg, drops, cloud, animations} = artwork(reduced);
    const scene = {svg, timers: [], motions: [], observer: null};
    shower = scene;
    function position() {
      if (shower !== scene) return;
      if (!anchor.isConnected || !anchor.getClientRects().length) { cancel(); return; }
      const box = anchor.getBoundingClientRect();
      const vp = window.visualViewport;
      const left = vp?.offsetLeft || 0, top = vp?.offsetTop || 0;
      const width = Math.min(vp?.width || innerWidth, Math.max(68, Math.min(116, box.width)));
      const height = Math.min(vp?.height || innerHeight, Math.max(60, Math.min(110, box.height * .86)));
      const right = left + (vp?.width || innerWidth), bottom = top + (vp?.height || innerHeight);
      const x = Math.min(Math.max(box.left + box.width / 2 - width / 2, left), right - width);
      const y = Math.min(Math.max(box.top - height * .41, top), bottom - height);
      Object.assign(svg.style, {left: x + "px", top: y + "px", width: width + "px",
        height: height + "px", visibility: "visible"});
    }
    scene.position = position;
    scene.checkVisibility = () => { if (document.hidden) cancel(); };
    document.body.append(svg);
    position();
    if (shower !== scene) return false;
    scene.motions = animations.map(play => play());
    if (typeof ResizeObserver !== "undefined") {
      scene.observer = new ResizeObserver(position);
      scene.observer.observe(anchor);
    }
    window.addEventListener("resize", position, {passive: true});
    document.addEventListener("scroll", position, {capture: true, passive: true});
    window.visualViewport?.addEventListener("resize", position, {passive: true});
    window.visualViewport?.addEventListener("scroll", position, {passive: true});
    document.addEventListener("visibilitychange", scene.checkVisibility);
    window.addEventListener("pagehide", cancel);
    function later(delay, fn) {
      scene.timers.push(setTimeout(() => { if (shower === scene) fn(); }, delay));
    }
    later(openingDelay, () => {
      position();
      if (shower === scene && !document.hidden) onOpen?.();
    });
    later(showerDuration, () => {
      svg.dataset.phase = "clearing";
      drops.remove();
      for (const motion of scene.motions) motion.cancel();
      if (!reduced) scene.motions.push(cloud.animate([{opacity: 1}, {opacity: 0}],
        {duration: clearingDelay, fill: "forwards"}));
    });
    later(showerDuration + clearingDelay, () => { cancel(); onStop?.(); });
    return true;
  }

  window.AiraRain = Object.freeze({start, cancel, get active() { return shower !== null; }});
})();
