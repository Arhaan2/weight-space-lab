import unittest

from weight_space_lab.evaluation import compare_paired, validate_records


def row(item, seed=0, correct=False, status='ok', **extra):
    return dict(item_id=item, seed=seed, correct=correct, status=status, **extra)


class EvaluationTests(unittest.TestCase):
    def compare(self, baseline, candidate, **kwargs):
        return compare_paired(baseline, candidate, bootstrap_samples=100, **kwargs)

    def test_explicit_schedule_keeps_missing_failures_in_denominator(self):
        schedule = [dict(item_id=item, seed=0) for item in ('a', 'b')]
        report = self.compare([row('a', correct=True)], [], schedule=schedule)
        self.assertEqual(report['baseline']['mean'], 0.5)
        self.assertEqual(report['candidate']['mean'], 0)
        self.assertEqual(report['candidate']['imputed_missing_records'], 2)
        with self.assertRaises(ValueError):
            self.compare([row('a')], [])
        with self.assertRaises(ValueError):
            self.compare([row('outside')], [], schedule=schedule)
        with self.assertRaises(ValueError):
            self.compare([], [], schedule=schedule + schedule)

    def test_record_validation_requires_booleans_finite_scores_and_unique_cases(self):
        for record in (row('a', correct=1), row('a', seed=True), row('a', score=float('nan')),
                       row('a', score=float('inf')), row('a', score=True), row('a', score=2),
                       row('a', correct=True, status='timeout'), row('a', extra='unknown')):
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_records([record])
        with self.assertRaises(ValueError):
            validate_records([row('a'), row('a')])

    def test_truncation_and_completion_metrics_are_distinct(self):
        base, candidate = [row('a')], [row('a', correct=True, status='truncated')]
        self.assertEqual(self.compare(base, candidate)['delta'], 1)
        self.assertEqual(self.compare(base, candidate, truncation_policy='fail')['delta'], 0)
        self.assertEqual(self.compare(base, candidate, metric='completion_within_budget')['delta'], 0)
        for status in ('error', 'timeout', 'missing'):
            self.assertEqual(self.compare(base, [row('a', status=status)])['delta'], 0)

    def test_unequal_repeats_weight_items_equally(self):
        base = [row('a', seed=i) for i in range(3)] + [row('b')]
        candidate = [row('a', seed=i, correct=True) for i in range(3)] + [row('b')]
        report = self.compare(base, candidate)
        self.assertEqual(report['candidate']['mean'], 0.5)
        self.assertEqual(report['candidate']['case_mean'], 0.75)
        self.assertEqual(report['paired_flips']['improved'], 1)
        self.assertEqual(report['case_changes']['improved'], 3)

    def test_bootstrap_clusters_repeats_and_is_reproducible(self):
        base = [row('a'), row('b', correct=True)]
        candidate = [row('a', correct=True), row('b')]
        report = self.compare(base, candidate, seed=7)
        repeated = lambda rows: [dict(r, seed=s) for r in rows for s in range(4)]
        other = self.compare(repeated(base)[::-1], repeated(candidate), seed=7)
        self.assertEqual(report['confidence_interval'], other['confidence_interval'])
        self.assertFalse(self.compare(base[:1], candidate[:1])['confidence_interval']['informative'])
        graded = self.compare([row('a', score=0.2)], [row('a', score=0.6)], metric='score')
        self.assertAlmostEqual(graded['delta'], 0.4)
