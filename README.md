# Changlu Guo's academic homepage

Static site published from `main` using GitHub Pages.

## Daily citation updates

The **Update Google Scholar metrics** GitHub Actions workflow runs daily at 05:17 UTC and can also be started from the Actions tab with **Run workflow**. GitHub may delay scheduled runs.

It fetches the public Google Scholar profile once, updates `data/scholar.json`, commits the successful snapshot, and explicitly requests a Pages build. No API key or personal token is needed. The website reads this snapshot and shows the last successful refresh date alongside total citations, h-index, and i10-index. Existing citation badges use matching paper counts.

Only titles already listed in `publications.js` are eligible for citation updates. The workflow never adds papers or changes publication status, authors, or reviewer activities. New papers must be added deliberately.

If Google Scholar returns an error, a verification page, the wrong profile, or incomplete metrics, the workflow fails without overwriting the last successful snapshot. If the browser cannot load the snapshot, the dated figures in the HTML remain visible. This is a daily best-effort refresh, not a real-time feed; Scholar can restrict automated requests.

Run locally with Python 3.9 or newer (no third-party packages):

```sh
python3 -m unittest discover -s tests
python3 tools/update_scholar.py
```

The public profile request follows the `Allow: /citations?user=` route in [Scholar's robots.txt](https://scholar.google.com/robots.txt), without pagination requests. If Google changes access restrictions, leave the saved figures in place and investigate the failed workflow.
