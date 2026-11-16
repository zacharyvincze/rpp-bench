# Configs

Each JSON file here describes a benchmark sweep: global timing, a `defaults` matrix (backends × dtypes × layouts × batch sizes × image sizes), and a list of ops that inherit or override it. Pass one with `--config=config/<name>.json`.

| Config | Purpose |
|---|---|
| [smoke.json](smoke.json) | Fast sanity check that every op runs; not for measuring performance. |
| [example.json](example.json) | Moderate HIP sweep for day-to-day exploration. |
| [sweep_host.json](sweep_host.json) | Full HOST sweep. |
| [sweep_hip.json](sweep_hip.json) | Full HIP sweep. |
| [full.json](full.json) | Periodic performance reference: both sweeps combined into one run. |
