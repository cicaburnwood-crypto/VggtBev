# P2B 10-scene local visualizer

Run `./start_visualizer.sh` or double-click `Launch_Visualizer.desktop`.

The bundle runs entirely on this PC:

- Habitat-Sim and all ten scene/navmesh pairs are under `scenes/`;
- P2B-NLL epoch 10, step 14320 is under `checkpoints/`;
- the frozen VGGT-Omega checkpoint is also bundled;
- both Python environments are bundled under `runtime/`;
- the model runtime binds to `127.0.0.1:8896`;
- the interactive UI binds to `127.0.0.1:8894`.

The scene catalog was deterministically sampled with seed `20260804` from the
training manifest's validation split. It contains ten unique HM3D scenes and
has zero training-scene overlap. Some files originate from HM3D's upstream
`train` asset directory, but they are validation-only under this model's
scene-grouped split.

The UI includes RGB, simulator GT, fused semantic BEV, evidential confidence,
the Observed-Free Gate boundary, and WASD control. Selecting another
scene rebuilds Habitat and its collision map, which normally takes 15–30
seconds. The browser reconnects automatically.

Runtime options:

```bash
UI_PORT=8894 MODEL_PORT=8896 GPU_INDEX=0 ./start_visualizer.sh
NO_BROWSER=1 ./start_visualizer.sh
```

The launcher intentionally does not accept external Python environment paths.
It always uses:

```text
runtime/ml-gpu/bin/python
runtime/habitat-sim/bin/python
```

All application code, model checkpoints, scene assets, Python packages, logs,
and PID files remain below this bundle directory. The host NVIDIA driver,
OpenGL stack, shell, and standard system utilities are still required.
The launcher re-executes itself with a minimal environment so paths inherited
from an IDE, ROS, Conda, or another project cannot enter the runtime.

Use `./stop_visualizer.sh` or press Ctrl+C in the launcher terminal to stop
both local processes.

Run `sha256sum -c CHECKSUMS.sha256` from the bundle directory to verify both
checkpoint files.
