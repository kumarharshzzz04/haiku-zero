Engine: TensorFlow 2.x + Keras 3 · Windows 11 + WSL2 · Custom BPE tokenizer (vocab 4096)Target hardware: RTX 4050 Laptop (6GB VRAM) · i5-13420H · ~6M parameter modelExpected training time: 1–3 hours on GPU

PROJECT GOAL
Build, train, and run HAIKU ZERO — a causal Transformer LLM built from scratch in Keras thatgenerates English haiku poetry. V2 replaces the GPT-2/tiktoken tokenizer (50,257 vocab = ~25M deadparameters on this corpus) with a custom-trained BPE (4,096 vocab), scales the network to 6 layers,and adds validation checkpointing, mixed precision, and an optional 5-7-5 meter enforcement module.

Why not native Windows TensorFlow: TF dropped Windows GPU support after v2.10. WSL2 is mandatoryfor the RTX 4050 to be used. Do not attempt native Windows TF GPU, do not use tensorflow-directml.

PHASE 0 — ENVIRONMENT SETUP (before any code)
STEP 0.1 — Install WSL2 (PowerShell as admin, run once):

wsl --install
Reboot. Create Ubuntu username/password.

STEP 0.2 — NVIDIA driver: The Windows NVIDIA driver is sufficient. Do NOT install CUDA toolkitmanually inside WSL.

STEP 0.3 — Python environment (inside Ubuntu terminal):

sudo apt update && sudo apt install -y python3-venv python3-pippython3 -m venv ~/.venvs/haiku && source ~/.venvs/haiku/bin/activatepip install "tensorflow[and-cuda]" tokenizers datasets pandas numpy syllables
STEP 0.4 — GPU verification (MUST pass before anything else):

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
Must print a GPU. If empty → update the NVIDIA driver on the Windows side.

STEP 0.5 — Agent working rules:

Project lives in ~/haiku_zero (Linux filesystem, NOT /mnt/c — 10x faster I/O).
The coding agent must run inside the WSL2 terminal (VS Code → "WSL" window → terminal).
If the agent runs on Windows, wrap every command:wsl -d Ubuntu -- bash -lc "cd ~/haiku_zero && source ~/.venvs/haiku/bin/activate && python ..."
DIRECTORY STRUCTURE
haiku_zero/├── data/│   ├── raw/                # downloaded Kaggle/HF files (any format)│   └── haikus_clean.txt    # canonical output of Module 1├── checkpoints/            # best.keras + last.keras├── backend/│   ├── prepare_data.py     # Module 1: merge, clean, normalize all sources│   ├── train_tokenizer.py  # Module 2: custom BPE training│   ├── dataset.py          # Module 3: encoding + tf.data pipeline│   ├── embedding.py        # Module 4: token + positional embeddings│   ├── multi_head_attention.py  # Module 5: causal multi-head attention│   ├── transformer_block.py     # Module 6: Pre-LN decoder block│   ├── model.py            # Module 7: full HaikuZeroLLM│   ├── train.py            # Module 8: training engine│   ├── generate.py         # Module 9: inference engine│   └── meter_check.py      # Module 10: optional 5-7-5 enforcement├── tokenizer.json          # trained BPE artifact└── PLAN.md                 # this file
DATASET SOURCES (how to download)
Source 1 — Kaggle (~86k English haiku):

Free account at kaggle.com
Datasets → search "haiku" → download the ~86k English haiku dataset (and any other large Englishhaiku CSV found — more is fine, Module 1 filters everything)
Unzip into data/raw/
Source 2 — Hugging Face (~75k curated English haiku):

python3 -c "from datasets import load_dataset; d = load_dataset('statworx/haiku'); d['train'].to_csv('data/raw/hf_haiku.csv')"
If the dataset name changed, search huggingface.co/datasets for "haiku" (English, 3-line).

Source 3 (optional) — GitHub: search "english haiku dataset" for plain-text corpora. Drop intodata/raw/.

Different formats/column names are expected. Module 1 normalizes everything.

PHASE 1 — DATA PIPELINE & CUSTOM TOKENIZER
Module 1: Data Preparation (backend/prepare_data.py) — STEP 1.0
Read EVERY file in data/raw/ (CSV, JSON, TXT — sniff format, adapt to column names). For eachcandidate haiku:

Normalize: lowercase, strip whitespace, collapse multiple spaces.
Keep ONLY poems with exactly 3 non-empty lines.
Reject lines containing URLs, @mentions, >60 chars per line, or >50% non-ASCII characters.
Deduplicate globally: key = poem text with punctuation/whitespace removed, casefolded.
Write canonical format to data/haikus_clean.txt, one haiku per 4 lines:
line1line2line3<eos>
Print final haiku count and token-length statistics. Target: 100k–300k haiku.
Module 2: Custom BPE Tokenizer (backend/train_tokenizer.py) — STEP 2.0
Use HuggingFace tokenizers (NOT tiktoken — tiktoken cannot train custom vocabularies).

models.BPE() with pre_tokenizers.ByteLevel(add_prefix_space=False) and decoders.ByteLevel().
Train on data/haikus_clean.txt withBpeTrainer(vocab_size=4096, special_tokens=["<pad>", "<bos>", "<eos>"]).Newlines survive as regular tokens (ByteLevel handles this natively).
Save to tokenizer.json.
Print sanity checks: encode("autumn moonlight") → decode round-trips exactly; "" is asingle ID; average tokens per haiku ∈ [15, 40].
Constant: VOCAB_SIZE = 4096 (including specials).
Module 3: Dataset Pipeline (backend/dataset.py) — STEP 3.0
Read haikus_clean.txt, encode to one integer array with the trained tokenizer.
Split BEFORE packing: shuffle at haiku level, 97% train / 3% val. Never split by window(prevents train/val leakage).
Pack each split into a continuous token stream; chunk into windows of seq_length + 1 = 65.
Emit pairs: X = seq[:-1], Y = seq[1:].
Return tf.data.Dataset with .shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE).
Print token totals per split. Sanity: initial loss must start near ln(4096) ≈ 8.32.
PHASE 2 — HYPERPARAMETERS
Hyperparameter	Value	Notes
vocab_size	4096	Custom BPE (Module 2)
seq_length	64	Never truncates a haiku, leaves prompt room
embed_dim	256	
num_heads	8	32 dims per head
num_layers	6	Freed budget from vocab fix
ffn_dim	1024	GELU activation (not ReLU)
dropout	0.1	On attention output + FFN output
batch_size	128	Fits 6GB with mixed precision
Optimizer	AdamW	weight_decay=0.01, clipnorm=1.0
LR schedule	CosineDecay	warmup_target=3e-4, warmup_steps=100, decay to end of training
Epochs	20–30	Keep BEST val-loss checkpoint, not the last
Precision	mixed_float16	float32 lm_head rule in Module 7 is MANDATORY
Total params	~6M	lm_head TIED to embedding matrix
PHASE 3 — MODEL ARCHITECTURE
Module 4: Embeddings (backend/embedding.py) — STEP 4.0
Token Embedding: (4096) → 256D.
Learned Positional Embedding: (64) → 256D.
Output = element-wise sum. Use keras.ops inside custom layers (not raw tf.*).
Module 5: Causal Multi-Head Attention (backend/multi_head_attention.py) — STEP 5.0
Hand-rolled (this is the learning goal — do NOT use keras.layers.MultiHeadAttention):

Dense projections for Q, K, V. Reshape to (batch, 8, 64, 32).
Scores = (Q · Kᵀ) / √32. Causal mask: upper-triangle set to -1e9 before softmax.
Softmax → weighted sum of V → concat 8 heads → Dense back to 256 → dropout 0.1.
Module 6: Pre-LN Transformer Block (backend/transformer_block.py) — STEP 6.0
Sub-layer 1: x → LayerNorm → Attention → Add(x)
Sub-layer 2: x → LayerNorm → FFN(Dense 1024, GELU → Dense 256) → Add(x)
Dropout inside each sub-layer output.
Module 7: Full Model (backend/model.py) — STEP 7.0
HaikuEmbedding → 6× TransformerBlock → final LayerNorm → lm_head.
lm_head is Dense(4096) with dtype="float32" — MANDATORY. float16 logits into softmax causeNaN loss under mixed precision. This is the #1 mixed-precision bug.
Tie weights: lm_head kernel = transpose of the token embedding matrix.
Print model.summary() — verify ~6M params. If >10M, vocab or weight tying is wrong.
PHASE 4 — TRAINING & INFERENCE
Module 8: Training Engine (backend/train.py) — STEP 8.0
keras.mixed_precision.set_global_policy("mixed_float16").
Loss: SparseCategoricalCrossentropy(from_logits=True).
Optimizer: AdamW(CosineDecay(...), weight_decay=0.01, clipnorm=1.0).
Callbacks:
ModelCheckpoint("checkpoints/best.keras", monitor="val_loss", save_best_only=True)
last.keras saved every epoch
CSVLogger("logs/training/training_log.csv")
HaikuPreview callback (custom, REQUIRED): every epoch, generate one sample haiku from theprompt "autumn moonlight" and print it. Primary debugging instrument.
Success criteria: train loss < 3.5, val loss < 4.5, previews become readable 3-line poems.If val loss rises while train falls → overfitting → raise dropout to 0.15 and retrain.
Module 9: Generation Engine (backend/generate.py) — STEP 9.0
Load checkpoints/best.keras + tokenizer.json.
Encode prompt with <bos> prefix; autoregressive loop up to 64 tokens.
Sampling: temperature 0.8 → top-p 0.9 (nucleus) → repetition penalty 1.1 on generated tokens.
Stop on <eos> or after 3 newlines. Decode with the ByteLevel decoder, print with line breaks.
Interactive REPL: type a seed phrase, get a haiku, repeat.
Module 10: Meter Check 5-7-5 (backend/meter_check.py) — OPTIONAL — STEP 10.0
Count syllables per generated line using the syllables library.
If not 5/7/5 → regenerate (up to 10 attempts), keep the closest attempt, flag as "near-haiku"if all attempts fail.
CLI flag: python -m backend.generate --strict.
AGENT EXECUTION CHECKLIST
[ ] Phase 0  WSL2 + venv + GPU visible (print devices before proceeding)[ ] data/raw/ populated (Kaggle + HF sources)[ ] backend/prepare_data.py     → haikus_clean.txt (report haiku count)[ ] backend/train_tokenizer.py  → tokenizer.json (report round-trip check)[ ] backend/dataset.py          → report token counts per split[ ] backend/embedding.py[ ] backend/multi_head_attention.py[ ] backend/transformer_block.py[ ] backend/model.py            → verify ~6M params in summary[ ] backend/train.py            → loss starts ≈ 8.3, previews printed each epoch[ ] backend/generate.py[ ] backend/meter_check.py      (optional)
DEBUGGING / FAILURE MODES (in order of likelihood)
GPU not visible in WSL2 → update NVIDIA driver on Windows, re-run Step 0.4.
NaN loss → lm_head float32 rule (Module 7.2) was skipped.
Garbage output → verify initial loss ≈ 8.3; check HaikuPreview improves epoch over epoch.
Val loss climbing → overfitting → dropout 0.15, retrain.
Encoding round-trip fails → ByteLevel pre-tokenizer/decoder misconfigured in Module 2.
EXPECTED LIMITATIONS
The model does not count syllables (transformers see tokens, not sounds). Lines will feelhaiku-like; Module 10 enforces strict 5-7-5 at generation time.
~2–8M training tokens is small — expect heavy stylistic mimicry rather than novelty.
CONTEXT RECOVERY PROTOCOL
If the coding agent or a context window drifts, quote:

"Refer to HAIKU ZERO v2 Blueprint (PLAN.md). We are on Step [X.0] in backend/[file.py].TensorFlow/Keras 3 on WSL2, custom BPE vocab 4096, tied embeddings, mixed_float16 with float32lm_head. Continue."


---

## AMENDMENT (runtime swap, approved)

The project is built on **PyTorch + native Windows CUDA**, not TensorFlow/Keras 3 + WSL2.
Reason: TensorFlow dropped native-Windows GPU support after 2.10, so TF would have required a
WSL2 install (admin + reboot). PyTorch uses the RTX 4050 directly. Architecture, tokenizer,
hyperparameters, training schedule, checkpoints, and success criteria are unchanged from this
plan; only the framework and the module internals change (keras.ops -> torch.nn, keras callbacks
-> the loop in backend/train.py, .keras -> .pt). See RUNBOOK.md section 7 for the full deviation
table, including two deliberate correctness fixes:

1. `initial_alphabet=ByteLevel.alphabet()` in the BPE trainer - without it, bytes unseen during
   training are dropped at encode time and generation silently loses characters.
2. Case is preserved (PLAN.md said lowercase); `--lowercase` restores the original behavior.

Module list is unchanged, with three additions: `backend/config.py` (all hyperparameters in one
place), `tests/test_smoke.py` (CPU end-to-end check), `RUNBOOK.md` (commands + expectations).
