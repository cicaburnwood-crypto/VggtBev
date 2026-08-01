# Four-scene Docker smoke data

This package contains four five-frame sessions from four unique HM3D scenes in
the frozen `data_build_unlimited_method1_v4.json` source split:

| Scene | Packaged session |
|---|---|
| `hm3d:00000-kfPV7w3FaU5` | `GPU1-5/session_001043_hm3d_00000-kfPV7w3FaU5` |
| `hm3d:00001-UVdNNRcVyV1` | `GPU2-4/session_000710_hm3d_00001-UVdNNRcVyV1` |
| `hm3d:00002-FxCkHAfgh7A` | `GPU1-1/session_001325_hm3d_00002-FxCkHAfgh7A` |
| `hm3d:00003-NtVbfPCkBFy` | `GPU2-3/session_000402_hm3d_00003-NtVbfPCkBFy` |

The image build copies only camera RGB, the four Method I BEV targets,
`metadata.json`, `ground_truth_trajectory.jsonl`, and `COMPLETE`. Simulator
depth, estimated/GT camera files, previews, and visualization artifacts remain
outside the image. A new immutable 3-scene train / 1-scene validation manifest
is generated inside the image.
