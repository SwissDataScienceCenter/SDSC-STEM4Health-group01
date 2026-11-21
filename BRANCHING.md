# Branch, PR and review conventions

## Branches

| Branch | Purpose |
|---|---|
| `main` | Submission-ready. Merged from `dev` only. |
| `dev` | Integration. All work lands here via PR. |
| `<name>/<topic>` | One piece of work, e.g. `crispin/mape-splits`. Lives hours, not days. |

```bash
git switch dev && git pull
git switch -c <name>/<topic>
git push -u origin <name>/<topic>
gh pr create --base dev
```

## Pull requests

- One topic per PR. Base: `dev`.
- Title: conventional subject (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`).
- Description: what changed and why, in two or three lines.
- Model changes add mean and worst-batch MAPE on both splits, before and after, from `src/evaluate.py`.

## Review

- `dev`: one approval from a teammate. `main`: two.
- Reviewer checks: scores come from `src/evaluate.py`, MAPE does not get worse, no data or credentials, notebook outputs cleared.
- Review within 30 minutes. The author merges with squash and deletes the branch.

## Rules

- Change `src/evaluate.py` or the splits only with team agreement.
- One owner per notebook. Notebooks import from `src/`.
- Freeze about 90 minutes before the deadline: only fixes merge, then `dev` → `main`.
