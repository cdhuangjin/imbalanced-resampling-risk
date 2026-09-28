"""Finite enumeration checking Theorem 1's ranking claim; not a new theorem."""
from itertools import permutations, product
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

p = np.array([.03, .12, .3, .6, .85])
labelings = []
mass = 0.
for bits in product([0, 1], repeat=5):
    y = np.array(bits)
    prob = np.prod(np.where(y == 1, p, 1-p))
    if 0 < y.sum() < len(y):
        labelings.append((y, prob))
        mass += prob


def expected_auc(score):
    return sum(prob * roc_auc_score(y, score) for y, prob in labelings) / mass


rank_scores = [(list(order), expected_auc(np.array(order))) for order in permutations(range(5))]
true_score = expected_auc(p)
monotone = expected_auc(p**2)
best = max(v for _, v in rank_scores)
assert np.isclose(true_score, best)
assert np.isclose(true_score, monotone)
out = dict(status='FINITE_ENUMERATION_PASS', posterior=p.tolist(), n_labelings=32, valid_labelings=len(labelings), score_orders=len(rank_scores),
           conditioned_on='test contains both classes; AUC otherwise undefined', conditioning_mass=float(mass),
           true_posterior_expected_auc=float(true_score), best_permutation_expected_auc=float(best), squared_posterior_expected_auc=float(monotone),
           interpretation='True posterior maximizes expected AUC for this example. Strictly increasing miscalibration preserves AUC. This is a finite numerical check, not a general proof or an empirical benchmark reproduction.')
target = Path(__file__).resolve().parents[1] / 'outputs' / 'theory_check.json'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
