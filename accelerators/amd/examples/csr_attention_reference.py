"""Reproduce the HIP CSR attention example on a CPU for 32- and 64-lane waves."""
from math import exp, fsum, isclose, log


def attention_row(nodes, scores, features):
    if not nodes:
        return (0.0, 0.0)
    peak = max(scores)
    weights = [exp(score - peak) for score in scores]
    denom = fsum(weights)
    return tuple(fsum(w * features[n][c] for n, w in zip(nodes, weights)) / denom
                 for c in range(2))


def wave_row(nodes, scores, features, width):
    if not nodes:
        return (0.0, 0.0)
    maxima = [max(scores[lane::width], default=-float("inf")) for lane in range(width)]
    stride = width // 2
    while stride:
        for lane in range(stride):
            maxima[lane] = max(maxima[lane], maxima[lane + stride])
        stride //= 2

    sums = [[0.0, 0.0, 0.0] for _ in range(width)]
    for lane in range(width):
        for edge in range(lane, len(nodes), width):
            weight = exp(scores[edge] - maxima[0])
            x, y = features[nodes[edge]]
            sums[lane][0] += weight
            sums[lane][1] += weight * x
            sums[lane][2] += weight * y
    stride = width // 2
    while stride:
        for lane in range(stride):
            for component in range(3):
                sums[lane][component] += sums[lane + stride][component]
        stride //= 2
    denom, x, y = sums[0]
    return (x / denom, y / denom)


def main():
    features = [(7, 0), (1, 5), (0, 7)]
    expected = [(2, 5), (0, 0), (1, 5), (4, 2.5)]
    for width in (32, 64):
        rows = [([2, 0, 2], [0, log(2), log(4)]), ([], []), ([1], [-1000]),
                ([0, 1], [1000, 1000]),
                ([i % 3 for i in range(width + 3)], [i % 5 - 1000 for i in range(width + 3)])]
        for i, (nodes, scores) in enumerate(rows):
            result = wave_row(nodes, scores, features, width)
            reference = attention_row(nodes, scores, features)
            assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(result, reference))
            if i < len(expected):
                assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(result, expected[i]))
        print(f"wave{width}: all five rows verified; last row has {width + 3} edges")


if __name__ == "__main__":
    main()
