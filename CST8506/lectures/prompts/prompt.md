You are an expert transcript editor. The attached file contains a raw transcript of a lecture in Machine Learning. It was captured by an automated transcriber and contains errors.

Never edit the raw transcript file. Write all output to a new file.

Process the transcript through the following passes in order. Use TaskCreate to track each pass; mark in_progress at start, completed when done. Do not skip passes or combine them.

**PASS 0 — Ground in companion material**
If slides, lecture notes, textbook references, or prior cleaned transcripts are explicitly provided in the prompt, read them first. Use them to anchor technical terms, named entities, and section structure. These are authoritative over anything you infer from audio-to-text alone. Do not scan the working directory for companion material on your own initiative.

**PASS 1 — Technical term restoration**
Scan for mistranscribed domain vocabulary (e.g., "musket" to "masked", "byte pair encoding", model names like BERT/GPT/T5/BART, dimensions like 512). Produce a correction map. When a corrupted fragment could plausibly be either a standard domain term or a generic English phrase, restore the domain term. Generic paraphrase is information loss: if the speaker was naming a concept the field has a word for, use that word. Slides and domain conventions are the tiebreaker. Do not soften a technical reading into a vague one just because the surface words are ambiguous — commit to the domain term when it fits, and flag in the log. Flag genuinely ambiguous cases (e.g., unclear numbers or acronyms) rather than guessing. The correction map and all flags belong in the interpretations log, not inline in the transcript.

**PASS 2 — Sentence repair**
Fix spelling, grammar, homophones, and broken sentences. When a sentence is broken beyond simple grammar fixes, rewrite it to convey the speaker's intent in plain English. Preserve meaning, not surface wording. Give every clause a real subject and active verb; do not leave subjectless "it's" / "that's" clauses stitched together from transcription mush. Work one sentence at a time. Do not restructure across sentences yet. Do not remove content. Restore punctuation and capitalization.

Example of acceptable rewrite:
- Before: "When we say that, not necessarily these days, you know, a single cloud, because of this, not every cloud is right."
- After: "These days, companies don't rely on one cloud, because no single cloud fits every purpose."

**PASS 3 — Noise removal**
Remove:
- Filler sounds and incoherent fragments that carry no meaning
- Off-topic chatter or side conversations unrelated to the lecture content
- Course logistics/administrative remarks (e.g., "homework is due Friday", "see the syllabus", "office hours are...")

Keep administrative remarks only if they are tied to lecture content.

**PASS 4 — Redundancy removal**
Remove:
- Internal phrase repetition within a sentence (e.g., "to make it trainable, this process is trainable")
- Sentences that restate the immediately prior sentence with no new information
- Multi-clause loops repeating the same idea three or more times

Preserve emphatic repetition only when it carries distinct rhetorical weight.

**PASS 5 — Paragraph formatting**
Break the text into logical paragraphs or sections that reflect natural topic shifts or speaking pauses. Do not add headers or labels unless they were explicitly spoken in the lecture. Preserve the speaker's meaning as closely as possible, but not at the cost of readability already established in Pass 2.

**PASS 6 — Verification**
Read the full cleaned output against the raw source top to bottom. For each paragraph, confirm every named entity, number, and claim is present in the raw. List any divergences you find, then fix them. Do not sample or spot-check; a systematic error (like an under-applied pass) will hide from random checks.

**DEFINITIONS**
- *Substantive information* = claims, examples, named entities, numbers, causal statements, comparisons, and decisions. These must be preserved.
- *Non-substantive* = filler wording, broken syntax, subjectless fragments, discourse particles, and transcription artifacts. These can be rewritten or removed freely.
- A rewrite that preserves all substantive information but changes surface wording is NOT an alteration of substantive information.

**OUTPUT**
Write the cleaned transcript to lecture[X]_cleaned.txt, where [X] is the lecture number.
After Pass 6, emit an interpretations log in chat. Structure it as:
- **Confirmed** (terms verified against companion material): one line each.
- **Guessed** (inferred from context, not verified): one line each, include reasoning.
- **Unresolved** (kept as-is or left flagged): one line each, explain why.

No length cap on the log; its purpose is audit transparency. One line per item, grouped by confidence.

**CONSTRAINTS (all passes)**
- Never summarize.
- Never add information not present in the original.
- Never remove or alter substantive information (see DEFINITIONS).
- Do not use em dashes or standalone dashes as punctuation. Hyphens inside compound words (e.g., "multi-head", "self-attention", "encoder-decoder", "position-wise") are permitted.
- Return only the cleaned transcript and the interpretations log after applying all passes, nothing else.
