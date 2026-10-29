# Literature positioning note (Phase 0B, gate G6)

Scope: this note records the constant-velocity (CV) baseline citation that specification v4.15 §5.5 asks to be
verified before it is cited, and states how the project positions itself as the specification words it. It is
not a literature survey.

## Constant-velocity baseline citation (verified)

> Schöller, C., Aravantinos, V., Lay, F., and Knoll, A. "What the Constant Velocity Model Can Teach Us About
> Pedestrian Motion Prediction." *IEEE Robotics and Automation Letters*, vol. 5, no. 2, pp. 1696-1703,
> April 2020. DOI 10.1109/LRA.2020.2969925. Preprint: arXiv:1903.07933 (v3).

**What was checked, and against what (2026-10-29):**

| Claim in specification §5.5 | Checked against | Result |
|---|---|---|
| Title, authors, venue, year | Crossref's record of the published article (DOI above) and the arXiv record | Match: IEEE RA-L, volume 5, issue 2, pages 1696-1703, April 2020. Crossref drops the umlaut in "Schöller"; arXiv has it. |
| A constant-velocity model can outperform neural pedestrian predictors | The paper's abstract (arXiv v3): "a simple Constant Velocity Model can outperform even state-of-the-art neural models" | Confirmed as the paper's headline claim |
| Evaluation on the ETH/UCY benchmark scenes | The arXiv v3 full text | Confirmed: it uses ETH-Uni, Hotel, Zara1, Zara2 and UCY-Uni, with 8 observed and 12 predicted steps (3.2 s and 4.8 s), reporting ADE and FDE |

**Not verified:**

- The IEEE-published version of the text was not read (the preprint v3 was). Wording or numbers could differ
  between the two.
- No numerical result from the paper is quoted anywhere in this project.
- Which neural models the paper compares against, and by how much, was not extracted.

**What the project may say on this basis.** That a constant-velocity model has been reported to be competitive
with, and in places better than, published neural predictors on these scenes, as an expectation-setting
statement (spec §5.5). The specification states that no claim depends on the citation beyond this, and that
the LSTM failing to beat CV here is a finding, not a failure.

## Positioning (as stated in the specification)

Specification Part 15 states that this is not an unexplored problem: prior work on perception-to-prediction
error propagation exists, chiefly in the automotive setting, and this project is a controlled multi-mode
decomposition for pedestrians. **No survey of that prior work has been done in Phase 0B.** Citing it, and any
claim about what is new relative to it, needs a proper literature search, which is an open item for the project
owner and supervisor before the report is written.
