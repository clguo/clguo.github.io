import json
import unittest
from update_visitors import merge_locations, parse_locations


class VisitorHistoryTests(unittest.TestCase):
    def test_older_sources_survive_a_rolling_source_window(self):
        old = {'name': 'Zurich, Switzerland', 'lat': 47.36, 'lon': 8.55}
        recent = {'name': 'Copenhagen, Denmark', 'lat': 55.67, 'lon': 12.56}
        self.assertEqual(len(merge_locations([old], [recent])), 2)
        self.assertIn(old, merge_locations([old], [recent]))

    def test_colocated_labels_are_not_discarded(self):
        a = {'name': 'Copenhagen, Denmark', 'lat': 55.67, 'lon': 12.56}
        b = {**a, 'name': 'Kobenhavn S, Denmark'}
        self.assertEqual(len(merge_locations([a], [a, b])), 2)

    def test_unknown_location_does_not_create_a_fake_ocean_marker(self):
        rows = [[-48, 28, 'Unknown Location', 14, 3], [47.36, 8.55, 'Zurich, Switzerland', 1, 1]]
        html = 'var start = "01/01/2026"; var end = "12/31/2026"; var addressPoints = ' + json.dumps(rows) + ';'
        _, _, locations = parse_locations(html)
        self.assertEqual([p['name'] for p in locations], ['Zurich, Switzerland'])

    def test_error_pages_do_not_replace_the_archive(self):
        for html in ['Access denied', 'var addressPoints = [];']:
            with self.assertRaises(ValueError):
                parse_locations(html)


if __name__ == '__main__':
    unittest.main()
