This is a privileged-information upper-bound diagnostic, not retrieval. The supplied causal program is an abstract, mechanically restructured clue from existing enrichment. It may be incomplete or wrong.

Do not reason from the clue alone. First inspect the visible candidate source, unified diff, changed symbols, repository context, and nearby tests. For each candidate, state in the rationale how a causal-program element plausibly maps to that visible evidence. Discard or weaken any clue element that the visible target does not support. The visible context remains available and is the primary source for concrete imports, APIs, call shapes, and assertions.

Generate materially different, target-specific hypotheses rather than paraphrasing the causal program. Instantiate its abstract preconditions, action sequence, lost property, and observable using only interfaces evidenced in the visible target context. Prefer the smallest test that distinguishes base/reference from candidate.

Each candidate must contain exactly one independently runnable pytest test. Construct all mutable application, router, model, configuration, and override state inside that test or its local helpers/fixtures. Do not rely on test ordering, shared mutable module state, module-level cleanup, or another generated candidate.

The checkout may contain an old library version. Use imports and compatibility patterns demonstrated by the supplied repository context; do not assume modern module paths, constructors, test clients, configuration syntax, or dependency behavior. Mentally verify that each test should pass on base/reference before returning it. Avoid assertions that merely fail on both versions.

Do not infer access to the hidden fix patch, developer regression test, exact reproduction steps, repository identifiers, exact constants, root-cause prose, fix strategy, bug-family label, or mechanism-primitive registry labels.
