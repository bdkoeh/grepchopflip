## Using GCF as a tool

For processing songs and selecting samples, start with `docs/INSTALL.md` and
`docs/AGENT_USAGE.md`. Use the checkout's `.venv/bin/gcf`; no repository development
workflow is needed merely to use the CLI. Humans judge musical usefulness.
The BSD workflow below applies when modifying the project.

## BSD project workflow

This project uses Bob Says Do (BSD). For substantial features and planned delivery, load the installed `bsd` skill and follow its workflow. Ordinary small fixes can be handled directly while keeping affected records accurate.

The owner keeps `.planning/` locally; it is intentionally excluded from Git. When those records are available, before continuing project work read `.planning/PROJECT.md`, `.planning/STATE.md`, and `.planning/state.json`. On a public checkout without those records, use the checked-in documentation and do not invent historical state. Load the roadmap, decisions, and slice documents relevant to the request. Reconcile recorded progress with actual code and verification; do not reset existing plans or assume recorded workers are still running.

Keep decisions, progress, pending work, and verification evidence updated after meaningful changes and before handing off. Distinguish implemented, verified, human accepted, and complete. Respect the recorded execution scope, pauses, and human checkpoints; this file does not authorize new work or bypass feedback.

If BSD is unavailable, explain that and use the existing project records for context without claiming to have loaded the skill. Preserve all other project instructions and surface any conflicts.
<!-- BSD:END -->
