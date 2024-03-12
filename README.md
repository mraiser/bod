# Bod
This command line app parses a directory of raw PNG files, finds the humans in the images, sorts them into individual directories and crops them into face and body shots.

```
USAGE: bod WORKDIR 0.6 256 6
where WORKDIR is a directory that contains a directory named raw containing png images
and 0.6 is the desired similarity threshold (0.0 is identical)
and 256 is the desired minimum size in pixels of at least one side
and 6 is the desired number of threads
```

Adapted from:
[https://github.com/huggingface/candle/tree/main/candle-examples/examples/yolo-v8](https://github.com/huggingface/candle/tree/main/candle-examples/examples/yolo-v8)

## Installation
Prerequisites: cargo, openssl, pkg-config, cuda, dlib, blas, lapack 
(On NixOs see default.nix)

Requires Rust:
[https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

    cargo build --release