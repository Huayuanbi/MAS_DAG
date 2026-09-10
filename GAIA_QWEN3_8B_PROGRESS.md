# GAIA tool adaptation and Qwen3-8B run progress

## Current tool coverage

The GAIA runtime registers 24 tools. In addition to web, document, spreadsheet,
archive, calculation, and media tools, this iteration added:

- RapidOCR image text and boxes;
- PyAV media stream probing;
- bounded FFmpeg video frame extraction;
- faster-whisper audio/video transcription;
- safe ZIP member materialization for chained readers;
- legacy XLS value/style reading;
- XLSX formula, fill, and font-color inspection;
- JSON/JSON-LD structured reading;
- PDB chain, residue, atom, coordinate, sequence, and distance analysis;
- generic colored spreadsheet-grid graph analysis.

Tool access remains role-scoped. Tool calls are node-internal ReAct events and
are stored in `node_tool_traces` with arguments, status, result, and latency.

## Validation and smoke tests

- Tool/converter tests: 9 passed.
- Real GAIA PNG OCR: 20 text regions extracted in about 1.76 seconds.
- Real GAIA MP3: 22.413 seconds, mono 44.1 kHz; faster-whisper produced a
  timestamped transcript.
- Synthetic H.264 video: probed correctly and three bounded frames extracted.
- Both validation ZIP tasks: XLS/XML and PDF/XLSX members materialized and read.
- Validation PDB and JSON-LD attachments: structured readers succeeded.

The five-task Qwen3-4B routing run completed 19/20 graphs before the context
margin fix, with 0 correct graphs, 101,581 summed tokens, and 109.84 seconds of
summed graph wall time. It exposed base-model routing and reasoning failures,
including malformed tool JSON and stopping after metadata inspection.

Two focused deterministic-tool regressions then succeeded:

- Colored spreadsheet traversal: `finalizer_only` answered `Yes` (wrong), while
  `primary_finalize` called style inspection followed by grid analysis and
  answered `No` (correct). The green graph has 49 vertices, bipartitions 25/24,
  and two degree-one vertices, independently ruling out a Hamiltonian cycle.
- PDB atom distance: `finalizer_only` was wrong, while `primary_finalize` used
  the PDB distance rounded to picometer precision and answered `1.456` (correct).

## Larger candidate run

`data/gaia/validation_balanced28_candidates_v2.json` contains 28 tasks and 208
candidate graphs: web 8, document 7, spreadsheet 6, and media 7. It includes
two additional direct-evidence topologies where primary output reaches the
finalizer without being available only through the verifier.

`run_gaia_qwen3_8b_balanced28.sh` starts an isolated Qwen3-8B vLLM server on a
verified idle GPU, resumes the 208-graph run, retains node outputs/tool traces,
and emits an aggregate report with `analyze_gaia_results.py`.

## Current external blocker

The unnecessary partial download under `/data/gzy/models/Qwen3-8B` was stopped
and deleted. The complete shared model at `/data1/yz/MAS_DAG/Qwen3-8B` has all
five expected safetensors shards (about 16 GB). The preferred runtime is
`/home/gzy/.conda/envs/vllm-cu124` with Torch 2.6.0+cu124, vLLM 0.8.5, and
Transformers 4.51.3; `/home/yz/.conda/envs/vllm` has the same versions as a
fallback. At the last check all four RTX 4090 cards were occupied by other
users' long-lived processes; no card had enough unreserved memory for a safe
BF16 8B deployment. No foreign process was stopped or modified.
