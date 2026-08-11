# Stage Contract: Literature Processing & Paper Intake (Layer 2)

Defines explicit inputs, processing workflow, and outputs for ingesting surgical computer vision papers into the literature base.

---

## 1. Contract Inputs

- **Layer 4 Working Artifact:** Raw PDF/text file in `papers/inbox/paper.pdf`, or an online paper reference (arXiv ID, DOI, URL).
- **Layer 3 Reference Template:** `papers/templates/extraction_prompt.md`
- **Layer 3 Code Convention:** `_config/code_conventions.md` (Section 1: Clean Shallow Cloning Protocol)
- **Layer 0 Identity Rule:** `GEMINI.md` (Rule 8: Factual Grounding & Zero Hallucination)

---

## 2. Standard Processing Workflow

### Step 1: Paper Sourcing & Reading
1. Open document from `papers/inbox/` or download online reference.
2. Read title, abstract, math formulations, and architectural claims.

### Step 2: Auto-Naming & Directory Setup
1. Formulate a concise directory slug (e.g. `2024_toponet_liver_landmarks`, `2024_gemap_depth_prompt`, `2022_mask2former`).
2. Create directory `papers/<concise_paper_name>/` and move/save `paper.pdf` inside.

### Step 3: Optional Repository Intake (`/repos/`)
If official code repository is cited:
```bash
git clone --depth 1 <paper_repository_url> repos/<repo_name>
rm -rf repos/<repo_name>/.git
```

### Step 4: Lossless Technical Extraction & Strict Grounding
1. Apply `papers/templates/extraction_prompt.md` to extract LaTeX equations, loss functions, tensor shapes, and hyperparameters.
2. Grounding Directive: Extract ONLY facts explicitly stated in the paper/code. Mark unstated parameters as `[Not Specified in Paper]`.
