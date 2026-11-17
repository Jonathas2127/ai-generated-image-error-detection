# Historical inference scripts

These three scripts preserve selected implementation variants from the original research. Credential assignments were replaced with `os.environ["GROQ_API_KEY"]`, the image directory now reads `IMAGE_DIR` (default: `images`), and comments were translated into English. They are included as implementation references, outside the supported offline evaluation package.

**Do not use them as a current quick start.** They retain historical top-level batch execution, prompt formats, parsing behavior and evaluation conventions, including `zero_division=1`. They still expect the original `images_labels.json` and private image files. Their two Llama 3.2 Vision preview model IDs were retired by Groq on April 14, 2025. [Official notice](https://console.groq.com/docs/deprecations).

| Script | Historical purpose |
| --- | --- |
| `debug.py` | Single vision-model classification and output inspection |
| `retry_response.py` | Vision-model question rephrasing and empty-label retry |
| `type_classification.py` | Vision analysis followed by text-model label classification |

The scripts historically imported `groq`, `scikit-learn`, `pandas`, `matplotlib` and `seaborn`; Excel output also requires a pandas-compatible writer. Exact historical package versions were not recorded. None of these dependencies are needed for the repository's offline quick start.

Repository preparation did not test API keys, send images to an API, retrain a model, or regenerate predictions. No current end-to-end model inference is claimed.
