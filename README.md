# youtuber

Pipeline for producing translated and dubbed documentary videos for YouTube.

Two pipelines, picked by `pipeline:` in the config or `--pipeline`:

- `doc` (main): take an existing public-domain documentary and translate it — `d1_source` (find and fetch),
  `d2_gpu` (transcription and vocal separation on a GPU machine), `d3_translate`, `d4_tts` (dubbing), `d5_subs`,
  `d7_assemble`, `d7b_bookend`, `d8_meta`, then publishing and cross-links (`d9*`), archiving (`d10`) and Shorts (`d11`).
- `original`: write an episode from a topic — `s1_topic` … `s9_publish`.

Stages live in `src/stages/`; each writes its output into the job folder and is skipped when that output exists.

```bash
python -m src.run --case <slug> [--from <stage>] [--to <stage>] [--force] [--config config.yaml]
python -m pytest
```

Configs: `config.yaml` (laptop), `config-zion*.yaml` (the GPU machine; see `docs/zion-deploy.md`).
YouTube authorization: `docs/youtube-oauth-setup.md`. Not in the repository: `secrets/` (OAuth tokens), `.env`,
`jobs/` (working files) and licensed music that may only be used inside videos.

Last change: 2026-10-12. Whether the model and machine settings in the configs are still current was not checked.
