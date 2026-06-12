![Ubuntu](https://github.com/pfeatherstone/hubert.cpp/actions/workflows/ubuntu.yml/badge.svg)

# hubert.cpp

C++ implementation of [DistilHuBERT](https://arxiv.org/pdf/2110.01900) using [Eigen](https://gitlab.com/libeigen/eigen).

## Example

```cpp
hubert::model net;

float audio[16000]; // 16kHz mono audio
std::span<const float> feats = net.encode(audio); // [T,768] : an array of 768 packed normalized features
```

## Features

- [x] Dynamic sizes (but no batches)
- [x] Weights compiled into the library
- [x] Block-based API
- [ ] Streaming API.

## Usage

- Useful for downstream tasks such as speech representation, speaker identification, classification, etc.
