# Dataset notes

The historical project annotated 90 AI-generated images collected from Reddit r/aiArt using search terms including "wrong" and "weird". This selection is exploratory and may introduce selection bias. Reddit descriptions do not independently verify the generation process of each image.

`annotations.json` preserves the author's original image-level labels with a normalized `labels` field. It has 90 unique filenames and 114 label assignments:

| Label | Images | Historical definition |
| --- | ---: | --- |
| wrong | 18 | Mistakes or errors, including object placement, impossible physics or factual inaccuracies |
| weird | 17 | Strange, unusual, unexpected, surreal or bizarre content |
| unrealistic | 45 | Fantastical, imaginary or otherwise implausible content |
| other | 34 | None of the above; exclusive of the other labels |

These definitions overlap and do not establish a validated annotation standard. The author-created reference annotations are not objective ground truth. No inter-annotator agreement was established.

No images are distributed. Original-post-to-filename correspondence and image redistribution permissions have not been established in a verified public manifest. File-level checksums of private source logs in `provenance.json` support local tracing; the logs themselves are not distributed, so independent verification of those hashes requires access to the originals.

Predictions are saved **parsed labels**, not fresh model outputs. Empty lists remain empty. Some predictions include `other` together with other classes; these are retained, scored, and counted as rule violations.

The four excluded mixed logs remain in the private original project. Their run boundaries have not been reconstructed. No first/last occurrence was selected automatically.
