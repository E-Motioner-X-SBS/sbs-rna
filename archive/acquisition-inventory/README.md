# `docs/acquisition-inventory` — archived

The acquisition tooling for the 833 GB corpus lived only on the
`docs/acquisition-inventory` branch, which was created **before the model
existed**. This folder is that branch's record, kept on `main` so the branch
itself can be retired without losing anything.

## Do not merge the branch

It is **47 commits behind `main`** and a merge would **delete 65 files**,
including all of `src/pharos/`. The PR its URL suggests
(`.../pull/new/docs/acquisition-inventory`) would propose exactly that
deletion. Its commits are recorded in `BRANCH_COMMITS.txt` and its full file
list in `BRANCH_FILE_LIST.txt`.

## What was taken to `main`, and where it went

| branch file | now on main at | changed? |
|---|---|---|
| `scripts/acquire_all.py` | same path | yes — a provenance note was added; the original is here as `acquire_all.branch-original.py` |
| `scripts/build_raw_pdb_entrylist.py` | same path | no |
| `scripts/watch_downloads.py` | same path | no |
| `pyproject.toml` | same path | no |
| `uv.lock` | same path | no |

Nothing else on the branch is absent from `main`. Every other difference is an
**older version** of a file `main` has since rewritten — 214 pinned checks
against today's 331, 92.65% ribosomal residues against the corrected 85.94%,
the superseded §5.4 sizing table. Those are not content to recover; they are
the numbers the later work replaced.

## Getting the datasets

`scripts/acquire_all.py` is the provenance record and the downloader: **147
jobs** across three groups, every source URL, every verified byte size,
resumable and idempotent.

```bash
uv run python scripts/acquire_all.py --list                 # the plan
uv run python scripts/acquire_all.py --group catalog --workers 12
uv run python scripts/acquire_all.py --group sequence --workers 10
uv run python scripts/acquire_all.py --only rfam --workers 4
```

**Read `--list` carefully on this machine before running anything.** The corpus
was downloaded on a different host and copied here; `data/` came across,
`data/acquisition/` did not. So 44 non-MARS jobs report pending while their
output sits on disk — a `cmd` job is "done" when
`data/acquisition/state/<name>.done` exists, and that directory is absent, and
an `http` job is "done" when its destination exists, and the tree was
reorganised after acquisition. Running a group to "fill gaps" here would
re-download hundreds of gigabytes into a second, parallel layout.

`scripts/sampling/audit_inventory_gap.py` measures the tree as it actually is
and is the authority: **18 sources, 17 present, 0 partial, 0 missing, 1 absent
by decision.** 833.3 GB documented, 567.0 GB on disk.

The one real absence is **MARS**: 30 jobs, 427.29 GB, skipped under decision
D18 because 1.73B sequences yielded 0.17% diverse structured ncRNA. To acquire
it anyway:

```bash
uv run python scripts/acquire_all.py --only mars --workers 8
```

That is the whole of the 266 GB difference between the documented inventory and
what is here. The other 17 sources exceed their documented size on disk,
because several are stored decompressed.
