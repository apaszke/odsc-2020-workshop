#include <torch/extension.h>

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <deque>
#include <tuple>
#include <memory>
#include <cstring>

struct StringView {
  StringView(const char* base, size_t length)
    : base(base), length(length) {}

  bool operator==(const StringView& other) const {
    return length == other.length && std::memcmp(base, other.base, length) == 0;
  }

  operator std::string() const {
    return std::string(base, length);
  }

  const char* base;
  size_t length;
};

namespace std {
template<>
struct hash<StringView> {
  size_t operator()(const StringView& str) const {
    size_t result = 0;
    constexpr size_t prime = 31;
    for (size_t i = 0; i < str.length; ++i) {
      result = str.base[i] + (result * prime);
    }
    return result;
  }
};
}

std::pair<std::vector<uint64_t>, std::vector<std::string>> tokenize(std::string path) {

  std::unordered_map<StringView, uint64_t> vocabulary {
    std::make_pair(StringView{"<eos>", 5}, 0)
  };
  std::vector<std::string> idx2word { "<eos>" };
  uint64_t next_token = 1;
  std::vector<uint64_t> tokens;

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data { new char[size] };
  file.read(data.get(), size);
  tokens.reserve(size / 4);

  size_t start = 0;
  size_t end = 0;
  while (end < size) {
    while (end < size && (data[end] != ' ' && data[end] != '\n')) { ++end; }
    StringView word(data.get() + start, end - start);
    auto it = vocabulary.find(word);
    if (it == vocabulary.end()) {
      auto token = next_token++;
      std::tie(it, std::ignore) = vocabulary.emplace(word, token);
      idx2word.emplace_back(word);
    }
    tokens.push_back(it->second);
    if (data[end] == '\n') {
      tokens.push_back(0);
    }
    while (end < size && (data[end] == ' ' || data[end] == '\n')) { ++end; }
    start = end;
  }
  return std::make_pair(std::move(tokens), std::move(idx2word));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_tokenize", tokenize);
}
