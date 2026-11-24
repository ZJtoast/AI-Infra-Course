/* Character size on the common 288 × 320 drawing grid. The reference is the
   standing cel's crown-to-chin span. Upright cels also use crown-to-sole height.
   A single measurement keeps a breathing loop steady. Six measurements follow
   drawings that change proportion as the character sits, turns or reaches. */
const AiraFraming = (() => {
  "use strict";
  const sampleFrames = [0, 2, 6, 12, 18, 24];
  const headSpan = {
    amused: 73,
    balance: 73,
    bell: [73, 89, 89, 89, 85, 90],
    bow: 73,
    breathe: 73,
    bubbles: 73,
    catch: [73, 73, 70, 67, 71, 73],
    chart: [97, 105, 108, 103, 109, 105],
    cheer: 73,
    curious: [73, 68, 68, 68, 68, 72],
    downcast: 73,
    fold: [94, 88, 88, 91, 91, 89],
    grateful: 73,
    hearthands: 73,
    idle_kneel: 91,
    idle_pipeline: 75,
    idle_prone: 97,
    idle_side: 95,
    idle_sit: 94,
    idle_sway: 73,
    idle_tokens: 91,
    inspect: [73, 77, 92, 103, 108, 106],
    jump: [69, 69, 69, 72, 72, 73],
    kvpage: 73,
    lace: [73, 80, 84, 96, 96, 94],
    land: 73,
    ledgehide: [67, 70, 75, 86, 93, 97],
    lift: [73, 84, 87, 79, 79, 85],
    matrixtiles: 73,
    nap: [94, 58, 58, 58, 58, 59],
    parasol: 73,
    peek: [73, 108, 103, 98, 98, 97],
    plane: [80, 80, 83, 78, 68, 72],
    poke: [73, 76, 79, 79, 79, 81],
    polish: [73, 73, 73, 67, 73, 78],
    pour: [91, 97, 102, 102, 102, 105],
    pout: 73,
    proud: [73, 73, 73, 70, 70, 69],
    react_kneel: 91,
    react_prone: [97, 94, 94, 91, 91, 97],
    react_side: [95, 84, 84, 84, 84, 95],
    react_sit: 94,
    reducepairs: [94, 98, 98, 104, 99, 96],
    relaychip: 73,
    relieved: 73,
    run: [73, 73, 86, 86, 91, 90],
    seal: [73, 73, 73, 73, 76, 76],
    shy: [73, 77, 77, 77, 82, 81],
    side_prone: [95, 92, 92, 95, 95, 97],
    sidehide: [74, 78, 82, 87, 90, 89],
    sip: [73, 69, 72, 72, 72, 73],
    sit_side: [94, 94, 94, 95, 95, 95],
    sketch: [73, 88, 99, 104, 109, 105],
    spin: 73,
    stack: [73, 80, 89, 89, 89, 92],
    starhug: [73, 73, 73, 79, 79, 73],
    startle: [73, 73, 73, 73, 70, 72],
    stretch: 73,
    study: [94, 94, 94, 91, 91, 91],
    sweep: [91, 91, 94, 94, 94, 95],
    think: 73,
    tinyheart: [73, 73, 76, 76, 76, 73],
    tokencard: 73,
    type: [94, 97, 101, 101, 101, 100],
    watch: 73,
    water: [73, 79, 87, 87, 87, 87],
    wave: 73,
    whisk: 73,
    winkpose: 73,
    yawn: 73
  };
  const reference = headSpan.breathe * 1.42;
  // The kick ends on the standing cel baked at 1.10. One fixed scale preserves
  // the camera movement throughout the film. The duster uses its neutral body cel.
  const films = {
    "maid-duster": {scale: 1.26, foot: [360, 581]},
    "maid-flying-squirrel": {scale: 1.42 / 1.10, foot: [360, 550]}
  };

  function scale(code, frame = 0) {
    if (films[code]) return films[code].scale;
    const spans = headSpan[code];
    if (typeof spans === "number") return reference / spans;
    const hi = sampleFrames.findIndex(at => at >= frame);
    if (hi <= 0) return reference / spans[hi === 0 ? 0 : spans.length - 1];
    const progress = (frame - sampleFrames[hi - 1]) / (sampleFrames[hi] - sampleFrames[hi - 1]);
    return reference / (spans[hi - 1] + (spans[hi] - spans[hi - 1]) * progress);
  }

  // Union of the visible atlas ink over all actions, relative to the common foot.
  const reach = {left: 216, right: 211, above: 465, below: 12};
  return {scale, films, reach};
})();
