# Does the published literature agree?

Every number in this project was measured against itself. Nothing in it had ever been
checked against the field. This document does that, claim by claim, because the obvious
question about a negative result — *is this real, or did you do it wrong?* — has no answer
anywhere else in the repo.

Searched 2026-09-20 via PubMed (bio-research MCP tools) and Consensus. **According to
PubMed**, and with DOI links to every source cited. This is a targeted check of *this
project's specific claims*, not a systematic review; §*What this review is not* states the
limits.

## The verdict

**The central result replicates.** The two largest studies using the same method class —
embedding a lesion mask in a normative connectome to infer disconnection — both find that
disconnection performs *at or below* lesion-based features, which is what Phase 3 found.

| this project's claim | literature | verdict |
|---|---|---|
| Disconnection ≈ location baseline | Salvalaggio 2020 (n=132): structural disconnection predicts "to a level **comparable to** lesion information" | **replicates** |
| Location ≥ disconnection | Khalilian 2025 (n=340): "lesion masks exhibited **superior** predictive performance relative to unthresholded" disconnection maps | **replicates, more strongly** |
| Lesion volume is the dominant term | Forkel 2014: lesion size the **only** independent predictor of left-hemisphere aphasia severity | **replicates** |
| Little survives removing volume | Medaglia 2022: after removing lesion volume, network topology gives "**modest** statistical value beyond lesion volume ... less value in predicting specific behaviors" | **partly** — they retain a small effect, this project retains none |
| Inferred functional disconnection is weak | Salvalaggio 2020: indirect functional disconnection "scarce or negligible" and no substitute for direct fMRI | not tested here (structural only) |

**Absolute performance is in the normal published range**, which matters as much as the
ordering. Phase 3's arms score R² 0.398–0.481 predicting WAB-AQ. Salvalaggio reports
0.16 < R² < 0.58 across eight behavioural domains from lesion and disconnection maps. This
project is not producing unusually weak models and then concluding from the weakness.

## What corroborates it

**Salvalaggio et al. 2020, *Brain*, n = 132.** The closest match to this project's design:
lesion masks embedded in healthy-subject atlases to derive indirect structural and functional
disconnection, then multivariate ridge against deficits in eight domains including language.
Conclusion: *"the indirect estimation of structural connectivity damage successfully predicted
behavioural deficits post-stroke to a level comparable to lesion information."* Comparable —
not better. That is Phase 3's answer, at half this project's sample size, five years earlier.
[DOI](https://doi.org/10.1093/brain/awaa156)

**Khalilian et al. 2025, *NeuroImage: Clinical*, n = 340.** The largest and most direct test.
Four lesion-derived feature sets compared — lesion masks, probabilistic structural
disconnection maps, connectome strengths, network topology — with PCA-ridge and LOOCV.
*"Lesion masks exhibited superior predictive performance relative to unthresholded"*
disconnection maps, and *"connectome-based features exhibited lower R² values"* still. The
ordering this project found is the ordering they found, at n = 340.
[[3]](https://consensus.app/papers/details/be8617c8ffdd56fa86745d346679743c/?utm_source=claude_code)

**Forkel et al. 2014, *Brain*.** Prospective tractography of perisylvian language networks
against WAB-R. In the left hemisphere, *"lesion size was identified as the only independent
predictor of longitudinal aphasia severity."* The project's most uncomfortable finding —
volume alone at R² 0.398, everything else fighting over 0.05 — is a decade old.
[DOI](https://doi.org/10.1093/brain/awu113)

**Medaglia et al. 2022, *J Neurosci*.** Independently important because they do the same
*control* this project does: remove lesion volume from both the network measures and the
WAB-AQ before testing. They conclude *"global network measures provide modest statistical
value beyond lesion volume when predicting overall aphasia severity, but less value in
predicting specific behaviors."* They retain ~10% of variance; this project retains none.
The difference is discussed below.
[DOI](https://doi.org/10.1523/JNEUROSCI.1163-21.2022)

## What contradicts it, and whether it should worry us

Three positive results exist. None of them is the same experiment.

**Del Gaizo et al. 2017, *eNeuro*.** Connectome lesion-symptom mapping *improved* prediction
of aphasia severity over traditional lesion-symptom mapping. The decisive difference:
**subject-specific diffusion MRI**, not a normative connectome. They measured each patient's
own surviving white matter. This project infers disconnection from an average brain, and
`docs/DISCONNECTION.md` limit 3 already names that as the weak point. This paper is evidence
that the *idea* works when you have the data for it — and indirect evidence that the
normative approximation is where the signal is lost.
[DOI](https://doi.org/10.1523/ENEURO.0204-17.2017)

**Kuceyeski et al. 2016, *Human Brain Mapping*, n = 40.** Normative-connectome disconnection
(the NeMo tool) predicting 6-month outcomes at R² 0.56–0.72, explicitly beating lesion-volume
models. This one *is* the same method class and does contradict. Two qualifications: n = 40
with outcomes in mobility and daily activity rather than language, and the comparison is
against **lesion volume alone** — not against a location baseline. `docs/BUGS.md` B1 is
entirely about how a weak baseline manufactures a win for disconnection, and volume-only is a
weaker baseline than the N=463 atlas this project races against.
[[2]](https://consensus.app/papers/details/bf03e80e592458029b80ea9d3447ed99/?utm_source=claude_code)

**Kristinsson et al. 2020 (n=116, [DOI](https://doi.org/10.1002/hbm.25321)) and Hu et al. 2025
(n=76, [DOI](https://doi.org/10.1016/j.neuroimage.2025.121300)).** Both beat lesion-only
baselines for WAB-AQ, and both do it by adding **new imaging modalities** — task fMRI,
resting-state connectivity, fractional anisotropy, cerebral blood flow. Hu states it plainly:
aphasia severity *"is influenced by factors beyond lesion location and volume."* Neither
contradicts this project, which asks a narrower question: given **one binary mask and nothing
else**, does re-representing it as disconnection beat re-representing it as location? Adding a
functional scan is a different and better-resourced experiment.

> A caution on reading these numbers against ours. Kristinsson and Hu report *r* between
> predicted and observed (0.53–0.70). Squaring that is not R²: correlation ignores calibration
> and is systematically the more flattering statistic. Phase 3's R² values are not directly
> comparable to them and are, if anything, held to a stricter standard.

## The one actionable finding

Khalilian 2025 reports that **thresholding** the disconnection maps — keeping only
disconnections present in a sufficient proportion of healthy subjects — *"significantly
improved predictive performance"*, with the best thresholds at 0.3–0.9 depending on the
domain. Unthresholded maps lost to lesion masks; thresholded ones were the best performers in
the study.

This project has never thresholded. It emits both matrices unfiltered by design, and
`docs/DISCONNECTION.md` reaches the same conclusion from the opposite direction — *"either
coarsen to networks, or restrict to edges where the two matrices agree"* — having found that
parcel-level edges anti-correlate between the two references at ρ = −0.318. **An external
group found independently that the unfiltered map is the wrong feature object.** That is the
single most concrete thing the literature says to try next, and it is cheap: the matrices are
already cached, the harness already takes arbitrary feature sets, and the counted matrix
already carries per-edge fibre counts to threshold on.

It is worth saying that this would still be tested against the same n = 228 with 95% fold
intervals ~0.4 wide, so a real +0.03 would remain invisible. It is a better feature object,
not a fix for the sample size.

## What this review is not

- **Not systematic.** Targeted searches against this project's specific claims, not a
  protocol-driven review with inclusion criteria. Publication bias runs against negative
  results, so the balance of published findings is not an unbiased estimate of the truth.
- **Not a method audit.** No paper here was read beyond its abstract and reported figures.
  Sample sizes, baselines and metrics are quoted as the abstracts state them.
- **Cohort differences are real and unexamined.** This project is chronic aphasia at a median
  721 days post-stroke; Salvalaggio is 2 weeks post-injury. `docs/PHASE3.md` already shows
  time post-stroke is inert *in this cohort* (partial r = −0.002) and would not be in an acute
  one.
- **Nothing here changes a measured number.** No result in `docs/PHASE3.md` was revised
  because of this document. The literature is context for the finding, not evidence about it.
