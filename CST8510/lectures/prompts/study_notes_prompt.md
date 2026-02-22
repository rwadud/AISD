# Study Notes Generator Prompt

Use the following prompt with any lecture transcript to generate HTML study notes.

---

```
Create comprehensive study notes in a single self-contained HTML file for the following lecture transcript. Follow this structure and style exactly:

**Layout & Styling:**
- Max-width 960px centered page with a clean sans-serif font
- CSS variables for a consistent color palette (accent blue, green, orange, red, purple, teal)
- Responsive design with a mobile breakpoint at 640px

**Structure:**
1. Centered header with lecture title, subtitle, and course name
2. Table of Contents as a numbered list with anchor links to each section
3. Sections with numbered h2 headings and h3 subheadings

**Visual Components (use where appropriate):**
- **Color-coded cards** (with left border) for key concepts, warnings, takeaways — use different colors to distinguish categories (e.g., red for risks/warnings, green for best practices, teal for datasets/tools, purple for methodologies, orange for caveats)
- **Grid layouts** for groups of related cards (2-3 columns)
- **Tables** with colored headers for comparisons and structured information
- **SVG diagrams** for pipelines, timelines, spectrums, and relationship maps
- **CSS flowcharts** (flexbox with arrow characters) for sequential processes
- **Score/progress bars** for any numerical results or metrics
- **Highlight boxes** (yellow background) for important notes and caveats
- **Badges** (pill-shaped) for quick categorical labels
- **Term spans** (highlighted inline) for key terminology

**Content Guidelines:**
- Cover every topic from the transcript — do not skip sections
- Include worked examples and specific numbers/scores mentioned in the lecture
- Capture student Q&A points where they add educational value
- End with a "Key Takeaways" section summarizing the most important points
- Keep language concise and study-oriented, not conversational

```
