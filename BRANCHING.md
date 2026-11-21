# Branch, PR and review conventions

## Branches

| Branch | Purpose |
|---|---|
| `main` | Submission-ready. Merged from `dev` only. |
| `dev` | Integration. All work lands here via PR. |
| `dev_<name>` | Your working branch, e.g. `dev_crispin`. |
| `<name>_<topic>` | Optional, for parallel work, e.g. `crispin_mape-splits`. Branch from `dev`, PR into `dev`, delete after merge. |

```bash
git switch dev_<name>
git pull origin dev        # stay in sync with dev
# work and commit
git push
gh pr create --base dev --head dev_<name>
```

## Pull requests

- One topic per PR. Base: `dev`.
- Title: conventional subject (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`).
- Description: what changed and why, in two or three lines.
- Model changes add mean and worst-batch MAPE on both splits, before and after, from `src/evaluate.py`.

## Review

- `dev`: one approval from a teammate. `main`: two.
- Reviewer checks: scores come from `src/evaluate.py`, MAPE does not get worse, no data or credentials, notebook outputs cleared.
- Review within 30 minutes. The author merges with a merge commit, not squash, so `dev_<name>` stays in sync with `dev`.

## Rules

- Change `src/evaluate.py` or the splits only with team agreement.
- One owner per notebook. Notebooks import from `src/`.
- Freeze about 90 minutes before the deadline: only fixes merge, then `dev` → `main`.
