# Plan — Cycle 5: buildability audit

## Why this framing

19 defects so far. The pattern across cycles 3-4 is that **derived quantities**
fail more often than measurements, and the newest category — #18 (attention head
count) and #19 (decoder size) — is **components never specified at all**, then
silently assumed small or free.

So cycle 5 asks one question of every component:

> **Could a competent engineer implement this from the specification alone,
> without inventing a number?**

Anything requiring invention is an unspecified component, and #19 showed those
are not small.

## Method

1. Enumerate every component the architecture needs to run end to end.
2. For each, locate its spec in ARCHITECTURE.md / configs / reference code.
3. Classify: SPECIFIED / PARTIAL / **MISSING**.
4. For MISSING items, check whether the gap hides a *circularity* — G6 found the
   router reads features produced downstream of itself. The motif bank is keyed
   on the interaction graph, which comes from the pair track: **same shape**.
   That check was never run on anything but the router.

## Step-locked order

| Step | Target |
|---|---|
| B1 | Enumerate components; classify SPECIFIED / PARTIAL / MISSING |
| B2 | Circularity sweep: which components consume outputs produced after them? |
| B3 | MoE expert width d_ff=128 at d=512 — is 0.25x below a useful floor? Never questioned |
| B4 | Loss weights lambda_1..5 — are any values specified? |
| B5 | Fix or explicitly defer every MISSING item; update all deliverables |
