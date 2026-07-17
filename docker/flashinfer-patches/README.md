# FlashInfer source recovery

`v0.6.8.post1-omniva-mnnvl-pre-allreduce-add.patch.gz` is the deterministic
gzip-compressed, mail-formatted patch series from FlashInfer base `00dd4470`
through reviewed commit `444738e8`. It contains the native MNNVL
pre-allreduce-add implementation, capability metadata, and tests used by this
image.

Reconstruct with:

```bash
git checkout 00dd4470
gzip -dc v0.6.8.post1-omniva-mnnvl-pre-allreduce-add.patch.gz | git am
```

The patch is copied into the image under
`/opt/flashinfer-gb200-phase65-prear-add/source/` and its SHA-256 is recorded in
the image labels.
