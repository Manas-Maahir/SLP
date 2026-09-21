# Presentation script

A slide-by-slide script for `Lesion_Disconnection_Status.pptx` (15 slides).
About 15 minutes, plus questions. Stage directions are in `>` blockquotes;
everything else is meant to be said more or less as written.

**The frame to hold for the whole talk:**

> I built the pipeline, I tested the idea properly, and the idea did not hold up.
> Here is how I know that, and here is why the way I checked matters.

Do not try to sell disconnection as a win. The point estimates rank it first, and
one of the slides exists specifically to show that ranking is unreliable. The
negative result, with this much checking behind it, is the stronger talk and it
is much harder to attack in questions.

---

## Slide 1 — Title (30 seconds)

> Do not read the slide. Say this and move on.

"This project asks whether we can predict how badly a stroke affects someone's
language by looking at which brain connections the stroke cut, rather than just
where it happened.

I'm going to show you the pipeline, the result, and the checks. The short version
is that the answer is no — and the interesting part is how confident I am about
that."

---

## Slide 2 — The question (1 minute 30)

"A stroke kills a region of brain tissue. The standard way to describe it is
*where* — which region died.

But a brain region is only useful because of what it's wired to. A lesion doesn't
just destroy the region, it also cuts the white-matter cables running underneath
it. So there's a competing idea: what matters isn't where the damage is, it's
which connections it severs.

That's a real, testable claim, and both sides of it can be measured from the same
lesion mask. Model A describes location — 463 anatomical regions, how much of each
one is damaged. Model B describes disconnection — which of 80 white-matter bundles
were cut, and which network-to-network connections were disrupted.

Then you train both on the same patients and compare."

> Point at the three outcomes on the right.

"I decided in advance what each result would mean, before running anything. That
matters, because if you decide afterwards, you'll find a way to like whichever
answer you get."

---

## Slide 3 — Current status (1 minute 30)

"Before the result, briefly — this is a working system, not a notebook.

A lesion mask goes in and a full report comes out: which anatomy is damaged, which
tracts are severed, two independent disconnection matrices, four figures, and a
predicted language score — or a refusal, if the lesion is outside what the model
was trained on.

It's been run over 108 real lesions with no QC exclusions, and the labelled cohort
is 228 patients who have both a lesion mask and a language score.

Two numbers to notice. The registration quality is 0.924 — that's a standard
overlap measure, and it's ordinary published quality. And there are 203 automated
tests, because most of what went wrong in this project was invisible without them."

> If the room is technical, add:

"Everything lives on one coordinate grid, and the code refuses to combine two
images unless it has verified they're on the same grid. That check exists because
early on I found two atlases that had identical dimensions and identical voxel
sizes but stored the x-axis in opposite directions. Merging them mirrored every
structure — a left-hemisphere lesion came back labelled right hemisphere. Nothing
about the shape or the file size would tell you."

---

## Slide 4 — Example output (1 minute)

"This is what one patient produces. Glass brain, the lesion on a template, the
severed tracts ranked, and the disconnection matrix.

Two things worth pointing out.

Everyone is a number. The original files arrived with patient names in the
filenames. The mapping from name to number is stored in one place that never
leaves the machine, and there's an automated check that greps every single file
this project writes, looking for patient identifiers. It has caught a real name
three times — twice because I typed one into a code comment while explaining
something.

Second, the tract list shows both the percentage severed and the absolute volume.
That's not decoration. If you rank purely by percentage, small tracts always win —
across all 108 patients, not one of the six largest tracts was ever ranked first,
even though the corpus callosum is involved in almost every large stroke."

---

## Slide 5 — Registration (1 minute 30)

"To compare patients you have to put them in a common space, and the labelled
scans arrived in the scanner's own coordinates — five different grids, two
different orientations, none aligned to a template.

The important decision is the second bullet. A lesion is a hole in the anatomy the
alignment is trying to match. If you don't tell the algorithm to ignore it, it
tries to fill the hole by pulling healthy tissue into it — and it distorts exactly
the region you're trying to measure. So the lesion is excluded from the similarity
calculation at every stage.

The last bullet is the check that actually matters. This is an aphasia cohort, so
essentially every lesion should be left-hemisphere. All 228 came back
left-dominant. If the alignment had flipped anyone, they'd show up on the right,
and I'd know."

> If asked why a picture and not just a number:

"I ran eight patients first and looked at them, before committing to the full
batch. The overlap score can look fine while something is visibly wrong."

---

## Slide 6 — Measurement reliability (2 minutes)

> This slide surprises people. Slow down.

"I built the disconnection measure twice, using two completely independent
sources. One infers connections from the shape and position of 80 known
white-matter bundles. The other counts actual reconstructed nerve fibres — 479,000
of them — and checks which ones pass through the lesion.

They share no data, no reference dataset, and no algorithm. So comparing them
tells you how much to trust either one.

At the level of broad brain systems, they agree — correlation of positive 0.33,
and they agree for 104 of 107 patients.

At the level of individual region-to-region connections, they *anti*-correlate.
Negative 0.32, and they agree for only 4 patients out of 108. The sign flips.

I tested three explanations and ruled them all out before accepting this. What's
left is resolution: the brain regions here average about the size of a sugar cube,
and the two methods keep picking different *neighbouring* regions inside the same
functional area. Both will say 'motor cortex to brainstem'; they disagree about
which specific patch of motor cortex.

The consequence is concrete: any claim about one specific connection being cut is
not trustworthy. Only the network-level summary is. That constrained everything I
did afterwards."

---

## Slide 7 — Main result (2 minutes 30)

> This is the core of the talk.

"Here's the comparison. Disconnection is at the top, location is in the middle,
lesion volume alone is at the bottom.

If you read only the bar lengths, disconnection wins and I have a paper.

Now look at the error bars. Those are 95% confidence intervals, and they span
about 0.4 — for every single model. They overlap almost completely.

Every model here was scored on exactly the same 25 cross-validation splits, so I
can compare them patient-by-patient rather than just comparing averages. When you
run the proper paired statistical test and correct for the number of comparisons,
nothing beats the location baseline. Nothing.

The only difference that's even nominally significant is against lesion volume,
not against the baseline the whole thesis is about — and it doesn't survive
correction either.

So the honest answer is: disconnection is not distinguishable from location on
this data."

> If someone says "but the ordering is consistent":

"It is, and that's exactly the trap. My own code originally reported that ordering
as the finding. I replaced it with the paired test, and the paired test says it's
a coin flip. Ranking the averages of distributions that overlap this much will
call noise a result every time."

---

## Slide 8 — Lesion volume (2 minutes)

> The uncomfortable slide. Do not rush it.

"This is the number that should bother you most, and it's why I trust the negative
result.

Lesion volume is one number. How many cubic millimetres of tissue are dead. You
get it by counting voxels in the mask — no atlas, no alignment to a template, no
tractography, nothing.

That one number explains about 40% of the differences in language outcome between
patients.

Everything else in this project — 463 anatomical regions, 80 tracts, two
disconnection matrices, 479,000 fibres — takes you from 40% to 48%.

In clinical terms: the best model is accurate to about 15 points on a 100-point
scale. Volume alone gets you 18. Patients differ from each other by 28 points on
average. So the whole sophisticated half of the pipeline is competing for the last
eighth of the result.

And it gets worse, because lesion volume can be reconstructed *from* the
disconnection features — up to 66% of it. Which means most of what those features
carry isn't connectivity. It's size, measured a more complicated way."

> The bottom-right card is worth one sentence:

"That 1.000 is a deliberate sanity check — a model given lesion volume, asked to
predict lesion volume, must score exactly 1. If it didn't, the evaluation code
would be broken and every other number on this slide would be meaningless."

---

## Slide 9 — Controlling for lesion volume (2 minutes)

"So I asked the sharper question directly. Take lesion volume out of the outcome
first, and then see whether anything predicts what's left.

The method matters. A model of lesion volume is fitted *inside each training
split* and subtracted, then the features are scored on the residual. If you fit
that model once across all patients — which is the obvious way to do it — you leak
information from the held-out patients into the thing you're predicting, and every
result inflates at once.

The answer is on the left. Every feature set lands at or below zero. Not just
disconnection — the location baseline too. And that holds for all 18 combinations
of feature set and model, not just the best one.

The red bar is the control. Lesion volume predicting its own residual scores
minus 0.015 — essentially zero, which is what correctly removed means. If that bar
were meaningfully above zero, it would mean the method leaked and the whole slide
would be void.

So the 0.05 on the previous slide isn't connectivity information sitting on top of
size. When size leaves, there's nothing underneath it."

> Be careful here if pressed:

"This doesn't prove there is no signal. It proves that at 228 patients, with these
methods, nothing predicts the volume-free residual. That's a statement about what
this study can detect."

---

## Slide 10 — Demographic covariates (1 minute)

"The obvious objection is that I've left out things that obviously matter — age,
sex, and how long ago the stroke was. They were sitting in the data file for all
228 patients, so this cost nothing to check.

They add nothing. On their own they score below zero. Added to lesion volume, they
make it slightly *worse*.

Time since stroke is the interesting one, because it should matter — people
recover. Its correlation with outcome, after accounting for lesion volume, is
minus 0.002. Effectively zero. The reason is at the bottom: this is a chronic
cohort, a median of about two years post-stroke. Recovery had already levelled off
before anyone was assessed. In a cohort measured in the first few weeks, it would
almost certainly matter.

And the footnote is a nice illustration of the whole problem. There *is* a
significant sex difference in outcome — until you account for lesion volume, at
which point it disappears. Women in this cohort happen to have smaller strokes.
Report it without the control and you've published a demographic finding that is
actually a lesion-size finding."

---

## Slide 11 — Cohort size (1 minute)

> This is a correction to my own work. Own it plainly; it lands well.

"My project plan said there were 902 sessions available and I was only using 228,
so getting more patients was the cheap way to improve the result.

That was wrong. 902 counts scanning sessions, not people. There are 245
participants. 228 have a lesion mask. The other 17 have a language score but no
mask, so they can't be used at all.

228 isn't a sample I chose. It's everyone. There is no more of this dataset to
get, and the larger stroke dataset that does exist doesn't release outcome scores.

It took one command to count the files. It should have been checked before it was
written into the plan."

---

## Slide 12 — Multiple-comparison correction (1 minute 30)

> Optional if you are short on time. Keep it if the audience is statistical.

"One more finding, which is about method rather than biology.

When you run many comparisons you have to correct the p-values for how many you
ran. What I found is that the corrected values depend on how many models happened
to be in the same command.

Same data, same measurement, same underlying test statistics — identical to four
decimal places. But run five extra combined models alongside, and the comparison
count goes from 7 to 15, and the headline p-value moves from 0.22 to 0.48.

I'm reporting this rather than fixing it, deliberately. The obvious fix is to
declare fewer comparisons — and that moves p-values in the direction that makes
results look better. That's exactly the wrong instinct. Nothing is significant at
either count, so it changes no conclusion here, but it's worth knowing that the
number is that sensitive to how you ran it."

---

## Slide 13 — Permutation test (1 minute 30)

> This is the slide that answers "how do you know any of this is real?".
> Point at the control row, not the top row.

"Everything I've shown you so far is a comparison between models. This slide asks
a more basic question: is any of it real at all?

The test is simple. Take the language scores, shuffle them randomly between
patients so the connection between brain and outcome is destroyed, and re-run the
whole thing. Do that two hundred times per model. If a model still appears to
work on shuffled data, it was never working.

None of them did. Every feature set scores around plus 0.43 to plus 0.51 on real
data, and around minus 0.21 on shuffled data. Not one shuffle in two hundred came
close to the real score — that's the 0.005, and it's the smallest number this test
can produce.

The row to look at is the last one. Age, sex and time since stroke score minus
0.13 on real data against minus 0.14 shuffled — indistinguishable. That's the
control, and it's deliberately in there. If demographics had come back looking
significant, I'd know the test was manufacturing results and I couldn't trust any
of the rows above it. It didn't, so I can.

One thing this does not say: it does not say disconnection beats location. Each
model is tested against randomness, never against another model. Four feature sets
all clearing the same bar is exactly what you'd expect when all four are largely
measuring lesion size."

---

## Slide 14 — The last experiment, and what's parked (1 minute)

"One thing was left that the literature specifically said to try: threshold the
disconnection maps, meaning keep only the well-supported connections and throw away the rest.
A study of 340 patients found that thresholded maps win where unthresholded ones lose.

I wrote down what I expected before running it, and I expected it to change nothing. It
didn't. The best threshold nudges the score up a couple of points, doesn't beat location once
you correct for multiple comparisons, and scores nothing once lesion volume is taken out.

The control is what settles it. I also ran the same number of connections chosen *at random*,
and random scored slightly higher than the carefully selected set. So the small gain came
from having fewer columns, not from picking the right connections.

That closes the modelling. Everything else on the old list, like graph neural networks and
embeddings, is a more elaborate way of representing the same map, and every representation
I've tried comes out as lesion size. What would change the answer is different data:
each patient's own diffusion scan, or a bigger labelled cohort. Not a cleverer model.

Parked with reasons: registration alternatives, lesion filling, and the viewer integration.
The hold-out is not spent, for the reasons on the previous slides."

---

## Slide 15 — Summary (1 minute)

"Where this leaves things.

On this cohort, at this sample size, lesion volume accounts for the predictive
performance. Neither disconnection nor location adds anything measurable beyond
it.

That's a negative result, and I want to be clear that I don't think it's a failed
project. The pipeline works, it's checked, and it produced a clear answer in the
direction I wasn't hoping for — after the point estimates initially suggested the
opposite. Catching that is the part I'd defend.

What would change the conclusion is a bigger labelled cohort, or a way of
representing disconnection that isn't mostly a proxy for lesion size. What would
*not* change it is a more complicated model on 228 patients with error bars this
wide."

---

## Anticipated questions

**"Isn't a negative result just a failed project?"**
No. The claim being tested is popular and plausible, and it's expensive to chase.
A properly checked negative saves other people from spending a year on it. The
value here is in the checking, not the outcome.

**"Why not use deep learning / a transformer?"**
228 patients, and between 80 and 463 features. A neural network would memorise 228
rows perfectly and generalise to nothing. More importantly, one single feature
already explains 40% and nothing else adds anything that survives — there's no
hidden structure for a bigger model to find. A graph neural network was in the
plan, gated on this test passing. It didn't pass.

**"Maybe your disconnection measure is just bad?"**
Possible, and slide 6 is me testing exactly that. I built it twice from
independent sources. They agree at network level, which is the level I used. If
both are wrong they're wrong in the same direction, which I can't rule out — but
they aren't obviously broken.

**"Isn't R² of 0.48 quite good for this field?"**
It's respectable, and that's the trap. 0.40 of it is available from lesion volume
alone. The question isn't whether the model predicts — it's whether disconnection
adds anything, and slide 9 says it doesn't.

**"What if the outcome measure is too coarse?"**
Fair. It's a single aggregate language score. A per-domain breakdown — naming,
comprehension, repetition — might localise better, and disconnection might matter
for one of them. That's a legitimate next study, and it needs the per-domain
scores, which I'd have to go and get.

**"How do you know the pipeline itself isn't broken?"**
Three ways. A positive control that must return exactly 1.000 and does. A test
that feeds the system pure noise and requires it to score zero or below. And 203
automated tests, including ones that deliberately corrupt an atlas and check the
safety checks fail.

**"Could you predict for a new patient today?"**
Yes, for a left-hemisphere stroke. The model is trained, saved, and wired into the
report. For anything else it refuses rather than returning a number — two-thirds
of my own 108-lesion set are right-hemisphere, and a language model has nothing
meaningful to say about those. It's a research tool, not a diagnostic.

---

## Pre-flight checklist

- Open the deck **before** the room fills; PowerPoint holds a file lock and a
  rebuild will fail while it's open.
- Know these six numbers cold: **0.398** (volume alone), **0.481** (best model),
  **228** (the whole cohort), **0.924** (registration quality), **−0.318 / +0.331**
  (the two-matrix disagreement), **1 of 6** (validation test progress).
- Slides 7, 8 and 9 are the argument. If you're cut to five minutes, show those
  three and slide 15.
- If asked something you don't know, say so and offer to check. This talk's whole
  credibility rests on not overclaiming.
