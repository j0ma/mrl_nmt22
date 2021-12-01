# Tests to write

- File loading works for src, tgt, both
    - [x] side: src, tgt
    - [x] ram: stream/load
    - [x] stream multiple times
- TSV loading works for both
    - [x] side: both
    - [ ] ram: stream/load
    - [ ] stream multiple times
- CorpusSplit
    - [x] Unify source and target [stack horizontally]
        - from_src_tgt
    - [x] Unify multiple rows in a single split [stack vertically]
        - stack_text_files

To enable all tests, make sure to tweak `FULL_DATA` variable in `test/test.py`.
Alternatively, it is possible to set it using the `MRL_FULL_DATA_PATH` environment variable.