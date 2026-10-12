# Capstone slideshow

The PDF, editable HTML slideshow, presenter notes, assets, and supporting results
are kept here. The application code stays in `../capstone_final_v3/`.

Run all four questions with `python -u run_presentation.py` from that application
folder. New reports are saved here under `presentation_runs/`.

Open `capstone-slides.html` in Chrome, Edge, or Safari. The file is self-contained
and does not need a server, installation, network connection, or API key.

- Use the left and right arrow keys or the Previous / Next buttons.
- Click **Speaker notes** (or press **N**) to open the narration in a separate
  window. Allow this file's pop-up if the browser asks. Notes update with the slide.
- Click **Present** (or press **F**) to hide controls and request full screen.
  Press **Escape** to exit. If full screen is unavailable, the controls still hide.
- **Print / PDF** opens the browser print dialog with one slide per page. Enable
  background graphics and disable browser headers/footers for the intended look.

For recording, capture only the slideshow window and keep the notes outside the
recorded area. Aim for nine minutes, including the slide 7 demonstration. Rehearse
first and check the exported recording.

This is an HTML slideshow, not a PowerPoint file. It uses the existing written
draft and saved observations. Update evidence and limitations if the system changes.

The presentation has twelve slides, with one metrics slide per question (slides 6–9).
All four result slides use the same fresh run, started October 12 at 3:13 p.m.
Central and completed at 3:16 p.m. Each question used the same settings:
1,500,000 total context characters, no per-article cap, three action rounds,
and up to six articles. Each request started with its own empty answer cache;
none was a cache hit. No agent settings were changed for this rerun.
`docs/latest-run-results.json` contains the reviewed four-question summary.
The raw report, retained evidence, structured log, and captured console output
are in `presentation_runs/20261012T201346826306Z/`.
Review found two complete supported answers, one incomplete comparison, and
one answer with an unsupported capital claim. These are four observations,
not an overall accuracy estimate. Chat cost totals about $0.49429 using the
configured rates and excludes unmeasured embedding costs.
Each result slide includes a verbatim AI response excerpt. The notes contain
short spoken sections and plain-language explanations. Narration rounds the
measurements for easier reading; the slides retain the exact figures. Full AI
answers are preserved in expandable reference panels and do not need to be read aloud. Code filenames and line references are omitted from the notes;
the separate log excerpts retain their source references.
The nine-minute timing plan includes an optional short demo on slide 7.
Generated illustrations appear on slides 1 and 10. Images are embedded in the HTML,
so the slideshow still works as one file. Separate image files and
generation prompts are in `assets/`.

The updated evaluation, result, context-limit, and next-improvement slides were
visually checked in Chrome. HTML structure, JavaScript syntax, and slide metrics
were checked against the saved report. Rehearse the notes and full-screen flow
before recording.

`log-excerpts.txt` provides the question, action steps, collected articles, stop
reason, full answer, and metrics for each of the four result slides. Open it in
a separate text-editor window while presenting. `log-excerpts.html` provides
the same material with navigation links and an **Open separate window** button.
The console-style text is reconstructed from saved structured log records,
not a captured stdout transcript; source file names and line numbers are included.
Regenerate both files without model calls using
`python3 presentation/build_log_excerpts.py` from the repo root.
The original local log files must be present to regenerate the excerpts.
