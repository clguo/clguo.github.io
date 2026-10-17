import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location('scholar', Path(__file__).parents[1] / 'tools/update_scholar.py')
scholar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scholar)

PROFILE = '''
<div id="gsc_prf_in">Changlu Guo</div>
<table id="gsc_rsb_st">
<tr><th></th><th>All</th><th>Since 2021</th></tr>
<tr><td><a>Citations</a></td><td class="gsc_rsb_std">1,155</td><td class="gsc_rsb_std">1147</td></tr>
<tr><td>h-index</td><td class="gsc_rsb_std">7</td><td class="gsc_rsb_std">6</td></tr>
<tr><td>i10-index</td><td class="gsc_rsb_std">6</td><td class="gsc_rsb_std">5</td></tr>
</table>
<table>
<tr class="gsc_a_tr"><td><a class="gsc_a_at">SA-UNet: Test Paper</a></td><td><a class="gsc_a_ac gs_ibl">757</a></td></tr>
<tr class="gsc_a_tr"><td><a class="gsc_a_at">Unlisted manuscript</a></td><td><a class="gsc_a_ac">3</a></td></tr>
<tr class="gsc_a_tr"><td><a class="gsc_a_at">New published paper</a></td><td><a class="gsc_a_ac"></a></td></tr>
</table>
'''


class ScholarTests(unittest.TestCase):
    def test_all_time_metrics_and_whitelist(self):
        data = scholar.parse_profile(PROFILE, ['SA UNet: test paper'])
        self.assertEqual((data['citations'], data['h_index'], data['i10_index']), (1155, 7, 6))
        self.assertEqual(data['publications'], {'saunettestpaper': 757})

    def test_blank_publication_count_means_zero(self):
        data = scholar.parse_profile(PROFILE, ['New published paper'])
        self.assertEqual(data['publications'], {'newpublishedpaper': 0})

    def test_challenge_and_wrong_profile_are_rejected(self):
        for html in ['<html>Verify you are human</html>', PROFILE.replace('Changlu Guo', 'Someone Else')]:
            with self.assertRaises(ValueError):
                scholar.parse_profile(html, ['SA-UNet: Test Paper'])

    def test_missing_metric_is_not_published_as_zero(self):
        with self.assertRaises(ValueError):
            scholar.parse_profile(PROFILE.replace('1,155', ''), ['SA-UNet: Test Paper'])

    def test_existing_snapshot_survives_network_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'scholar.json'
            output.write_text('{"citations":1155}')
            error = HTTPError(scholar.PROFILE_URL, 429, 'Too Many Requests', {}, None)
            with patch.object(scholar, 'urlopen', side_effect=error):
                with self.assertRaises(HTTPError):
                    scholar.update(output)
            self.assertEqual(output.read_text(), '{"citations":1155}')

    def test_existing_snapshot_survives_invalid_response(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'scholar.json'
            output.write_text('{"citations":1155}')
            with patch.object(scholar, 'urlopen') as response:
                response.return_value.__enter__.return_value.read.return_value = b'captcha'
                with self.assertRaises(ValueError):
                    scholar.update(output)
            self.assertEqual(output.read_text(), '{"citations":1155}')


if __name__ == '__main__':
    unittest.main()
