DAFormer: Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses.
Users should be careful about adopting these features in any commercial matters.

- SegFormer and MixTransformer: Copyright (c) 2021, NVIDIA Corporation,
  licensed under the NVIDIA Source Code License ([resources/license_segformer](resources/license_segformer))
    - [daseg/models/decode_heads/segformer_head.py](daseg/models/decode_heads/segformer_head.py)
    - [daseg/models/backbones/mix_transformer.py](daseg/models/backbones/mix_transformer.py)
    - configs/\_base\_/models/segformer*
- DACS: Copyright (c) 2020, vikolss,
  licensed under the MIT License ([resources/license_dacs](resources/license_dacs))
    - [daseg/models/utils/dacs_transforms.py](daseg/models/utils/dacs_transforms.py)
    - parts of [daseg/models/uda/dacs.py](daseg/models/uda/dacs.py)

This repository is based on MMSegmentation v0.16:
Copyright (c) 2020, The MMSegmentation Authors, licensed under the Apache
License, Version 2.0 ([resources/license_daseg](resources/license_daseg))
