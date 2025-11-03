Code for reproduce [InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](https://arxiv.org/abs/2506.15745v2) [NeurIPS 25]    

* Please note that this code is for research and reproduction purposes only and does not include all components of the full methodology described in the paper.
  In particular, vision encoding with block-prefill processing requires additional implementation.
* Supported compression methods:

  * `uniform`: uniform selection
  * `swa`: sliding window attention
  * `tar_val`: TaR + VaN hybrid
* `BLOCK_SIZE` determines the block granularity for CKV compression, and `COMPRESS_FRAME_NUM` controls the number of frames to compress in the KV cache.
* For faster evaluation, it is recommended to pre-dump and load intermediate outputs (see `--load_dumped` and example scripts).
* Contributions and extensions to this repository are always welcome.