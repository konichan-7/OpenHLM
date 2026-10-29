# G1 + Dex3 playback assets

These files were relocated without modification from
`src/GR00T-WholeBodyControl4OpenHLM/gear_sonic/data/robots/g1/`
in OpenHLM commit `c68c07a`.

Only `g1_29dof_with_hand_rev_1_0.urdf` and every mesh referenced by that
URDF are retained. They support the 43-actuated-joint Rerun playback; the
GR00T/SONIC controllers, checkpoints, other robot models and simulation scenes
are outside this reduced repository.

The original component license and third-party notices are preserved in
[legal/](legal/). Asset-specific terms remain those of the original providers;
relocation does not relicense these files.

Meshes use Git LFS. If they contain a `version https://git-lfs.github.com/spec/v1`
pointer instead of mesh data, fetch this directory with:

```bash
git lfs pull --include="assets/g1/meshes/**"
```
