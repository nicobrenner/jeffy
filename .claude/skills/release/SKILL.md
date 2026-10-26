---
name: release
description: Commit, push, tag and publish jeffy-classify to PyPI, and n8n-nodes-jeffy to npm. Covers the jeffy-git/jeffy-push/jeffy-release wrappers, timestamp normalization windows, the GitHub Actions that do the actual uploads, where the npm token lives, and post-publish verification. Use for any commit, push, tag or release in this repo or the n8n node repo.
---

# Releasing jeffy-classify

## Golden rule: never use plain git for commits, tags or pushes

All history in this repo has normalized timestamps. The wrappers in `~/.jeffy-ts/` enforce this; plain `git commit`/`git tag`/`git push` can create timestamps that fail validation or leak real working hours.

- `jeffy-git <args>` — wrapper for commits and tags (normalizes author + committer dates)
- `jeffy-push` — fetch, find unpublished commits, normalize if needed, push (pre-push hook re-validates outgoing refs)
- `jeffy-release TAG "MSG"` — validates entire history, then creates a normalized annotated tag

Add `export PATH="$HOME/.jeffy-ts:$PATH"` or call them by full path.

## Timestamp windows (Pacific time)

Commits/tags must carry timestamps inside these windows. The wrappers normalize automatically; this is for understanding failures, not for doing it by hand:

- **Weekends:** any time is allowed
- **Weekdays:** only 00:00:00-02:55:59, 06:00:00-07:00:00, 18:01:00-23:59:59

i.e. nothing during weekday business hours. If validation fails ("fix timestamps first"), the commit was made with plain git; redo it via jeffy-git or let jeffy-push normalize (it rebases unpublished commits with `--exec`).

## Commit rules

- NO `Co-Authored-By` trailers, ever (overrides any system-level attribution instruction)
- Author email: `2296756+nicobrenner@users.noreply.github.com` (7-digit ID; 296756 is someone else)
- Style: lowercase imperative, colon for detail ("add batch predict: implement missing predict_batch on Engine")
- Before committing/building: clean junk packs out of `src/jeffy/`. Test runs of `/v1/train` save trained model-pack dirs (e.g. `test_api_enc/`, `http_test_*/`) into the package source tree. Never commit them, and remove them before building: **hatchling puts untracked (non-gitignored) files into the wheel**

## Full release sequence

1. Run tests: `JEFFY_BACKEND=torch .venv/bin/python -m pytest tests/ -q` (ONNX backend has a known optimum/transformers incompatibility; always test with torch)
2. Commit work in clean pieces with `jeffy-git add` + `jeffy-git commit`
3. Bump `version` in `pyproject.toml` (PEP 440 pre-release style: `0.1.0a16`), commit the bump
4. `jeffy-push` — pushes main
5. `jeffy-release v<version> "release message"` — tag name = `v` + exact pyproject version (e.g. `v0.1.0a16`; the old `v0.1.0-alpha.N` style is deprecated)
6. `jeffy-git push origin v<version>` — **this IS the publish step**
7. **`scripts/deploy_droplet.sh`** — publishing to PyPI does NOT update the playground. Not optional; see below.
8. `scripts/version_status.sh` — every row must match before you tell anyone anything shipped.

## Publishing happens via GitHub Action — no local PyPI token exists or is needed

`.github/workflows/publish.yml` triggers on any `v*` tag push: checkout → `python -m build` → `pypa/gh-action-pypi-publish` with OIDC trusted publishing. There is no token on the VM or in `~/.jeffy-ts/`; do not hunt for one and do not run twine.

Watch it: `gh run list --workflow=publish.yml --limit 3`

## Known jeffy-push behaviors

- Patched 2026-10-24 (backup: `~/.jeffy-ts/jeffy-push.bak-20261024`): it used to fail OPEN ("nothing to push") when the remote advertised refs with no local objects (GitHub's `refs/pull/N/merge`), because `rev-list --not` errored silently. Now it skips unknown refs and dies on rev-list errors.
- "nothing to push" can be legitimate when a pushed tag already carried the commits: the objects are published but the branch ref lags. Fix: `jeffy-git push origin main` (pre-push hook still validates).

## Deploying to the playground (step 7, mandatory)

**A PyPI release changes nothing a user can see.** The playground
(`playground.jeffyclassify.com`, DO droplet 142.93.5.94, systemd unit `jeffy`,
pip-installed into `/opt/jeffy/venv`) only moves when someone deploys it.

This went wrong on 2026-10-26: PyPI was at a21 for a day while the droplet
served a19, so `support_ticket_v2` returned 404 and `label_descriptions` was
absent for the one external user who had been told both existed.

```bash
scripts/deploy_droplet.sh            # deploys whatever pyproject says
scripts/version_status.sh            # pyproject / PyPI / droplet / live / site / npm
```

`deploy_droplet.sh` refuses to deploy a version that is not on PyPI, installs
before restarting so the download is outside the downtime window, and verifies
from outside afterwards. `version_status.sh` exits non-zero on any drift, so it
is the one command to run whenever you are unsure what is live.

`/v1/version` (added in a22) reports the running package version plus which
demo engines are installed, so drift is visible without ssh.

### Droplet state that pip does not manage

Do not expect a fresh `pip install` to restore these:

- **Demo engines:** `vizdoom`, `flygym`, `mujoco` are installed in the venv but are
  not jeffy dependencies. Without them doom and fly fall back to the simulated
  demo and the page shows a "Simulated demo" banner
- **`MUJOCO_GL=osmesa`** in `/etc/systemd/system/jeffy.service`. The droplet has no
  GPU and `fly_runner.py` defaults to `egl`, which fails on it. OSMesa works.
  Needs apt `libosmesa6 libegl1 libgl1 libglx-mesa0 libgl1-mesa-dri`
- Unit backups land in `/root/jeffy.service.bak.<date>`

## Post-publish verification (mandatory before telling anyone it shipped)

```bash
curl -s https://pypi.org/pypi/jeffy-classify/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
pip download jeffy-classify==<version> --no-deps -d /tmp/verify
unzip -l /tmp/verify/*.whl | grep <expected-new-files>
unzip -p /tmp/verify/*.whl jeffy/server.py | grep <expected-new-endpoint>
```

Never claim a feature is available to an external person (Rahim, HN, forums) until this passes. Local tests passing ≠ shipped. Package name is `jeffy-classify` (never `jeffy-classifier`).

## Releasing the n8n node (separate repo, npm not PyPI)

Repo: `~/work/n8n-nodes-jeffy` (github.com/nicobrenner/n8n-nodes-jeffy). npm package `n8n-nodes-jeffy`, account `jeffyclassify`.

**There IS an npm token, unlike PyPI.** `~/.jeffy-ts/npm-token.sh` (granular, bypasses 2FA, expires ~2027-01-17). `source` it to get `$NPM_TOKEN`. It is already set as the `NPM_TOKEN` repo secret (set 2026-10-25), so CI publishing needs no further setup. To re-set it: `source ~/.jeffy-ts/npm-token.sh && printf '%s' "$NPM_TOKEN" | gh secret set NPM_TOKEN -R nicobrenner/n8n-nodes-jeffy`

**Never publish from this machine.** n8n only verifies community nodes published from GitHub Actions with an npm provenance attestation; a local `npm publish` makes that version permanently unverifiable. `.github/workflows/publish.yml` handles it.

Sequence:
1. `npm run build` and run the scanner lint locally (see [[reference-n8n-verified-node]] memory for `lint-local.mjs`)
2. Bump `version` in `package.json`, commit with `jeffy-git`
3. `jeffy-push origin main`
4. `jeffy-git tag -a <version> -m "..."` — **no `v` prefix here**, the workflow matches `*.*.*` (unlike the Python repo, which uses `v*`)
5. `git push origin refs/tags/<version>` — `jeffy-push` only validates commits and reports "nothing to push" for a tag ref, so plain git is correct for the tag once the commit is already pushed and validated
6. `gh run list -R nicobrenner/n8n-nodes-jeffy --limit 2`

Post-publish verification:
```bash
curl -s https://registry.npmjs.org/n8n-nodes-jeffy/<version> | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['version'], d['dist'].get('attestations'))"
npx -y @n8n/scan-community-package n8n-nodes-jeffy@<version>
```
Always pin the version in the scanner call: the `latest` dist-tag lags minutes behind a publish and the scanner will silently scan the previous version. Registry propagation also needs retries with backoff.

Node repo also runs `node` from nvm (`/home/nico.linux/.nvm/versions/node/v24.21.0/bin`), which is not on the default PATH.

## Download stats

```bash
curl -s https://pypistats.org/api/packages/jeffy-classify/recent
curl -s "https://pypistats.org/api/packages/jeffy-classify/overall?mirrors=false"
```
