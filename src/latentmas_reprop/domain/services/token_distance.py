"""Unit-cost Levenshtein distance between token sequences."""


def token_levenshtein(left, right):
    """Return exact edit distance using bit-parallel dynamic programming.

    Python integers hold a column's bits, keeping long readout comparisons cheap
    without adding a compiled dependency. Insertions, deletions and substitutions
    each cost one.
    """
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    matches = {}
    for index, token in enumerate(left):
        matches[token] = matches.get(token, 0) | (1 << index)
    positive, negative = ~0, 0
    distance = len(left)
    high = 1 << (len(left) - 1)
    for token in right:
        equal = matches.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & high) - bool(minus & high)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = minus | ~(vertical | plus)
        negative = plus & vertical
    return distance
