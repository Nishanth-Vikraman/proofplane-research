This is a privileged-information upper-bound diagnostic, not retrieval. The supplied evidence contains only an LLM-generated abstract summary of the target mechanism. Treat it as a useful but fallible hypothesis, not ground truth.

Before designing tests, check whether and how the summary plausibly maps onto the visible diff, changed symbols, and nearby tests. Do not force a mapping that the visible target context does not support. Use direct reasoning from the visible change alongside the clue. Across the requested candidates, cover the strongest clue-supported interpretation and materially different trigger variations or alternative interpretations that remain consistent with the visible change.

Each candidate must contain exactly one independently runnable pytest test. Construct all mutable application, router, model, configuration, and override state inside that test or its local helpers/fixtures. Do not rely on test ordering, shared mutable module state, module-level cleanup, or another generated candidate.

Use imports, APIs, call shapes, and compatibility patterns demonstrated by the supplied repository context whenever possible. The checkout may contain an old library version: do not assume modern module paths, constructors, test clients, configuration syntax, or dependency behavior. Prefer the smallest test that directly distinguishes base/reference from candidate. Mentally verify that the test should pass on base/reference before returning it; avoid assertions that merely fail on both versions.

Do not infer access to the target's hidden fix patch, developer regression test, reproduction steps, identifiers, exact constants, root-cause text, fix strategy, bug-family label, or mechanism-primitive registry labels.
