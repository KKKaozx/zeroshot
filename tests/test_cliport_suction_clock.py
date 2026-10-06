"""Clock boundaries and order used by the live suction timing experiment."""
import unittest
from diagnostics.replay_cliport_native import suction_schedule


class SuctionClockTests(unittest.TestCase):
    def test_boundaries_preserve_order_and_commands(self):
        requests = [dict(tick=t, open=o) for t, o in
                    [(0, True), (95, False), (96, True), (97, False)]]
        for mode, expected in [('exact', [0, 95, 96, 97]),
                               ('start', [0, 0, 96, 96]), ('end', [0, 96, 96, 192])]:
            result = suction_schedule(requests, mode)
            self.assertEqual([r['scheduled_tick'] for r in result], expected)
            self.assertEqual([r['open'] for r in result], [r['open'] for r in requests])
            self.assertEqual(len(result), len(requests))

    def test_invalid_clock_rejected(self):
        for requests in [[dict(tick=-1, open=False)], [dict(tick=1.5, open=False)],
                         [dict(tick=2, open=False), dict(tick=1, open=True)],
                         [dict(tick=1, open=0)]]:
            with self.assertRaises(ValueError):
                suction_schedule(requests, 'exact')
