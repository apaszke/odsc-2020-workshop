#include <torch/extension.h>

#include <utility>
#include <vector>

using LSTMState = std::pair<torch::Tensor, torch::Tensor>;

LSTMState lstm_cell(const torch::Tensor& input, const LSTMState& state,
                    const torch::Tensor& weight_ih, const torch::Tensor& weight_hh,
                    const torch::Tensor& bias) {
  torch::Tensor hx, cx;
  std::tie(hx, cx) = state;

  auto gates_vec = torch::mm(input, weight_ih.t()) +
                   torch::mm(hx, weight_hh.t()) +
                   bias;
  auto gates = torch::chunk(gates_vec, 4, 1);

  auto ingate = torch::sigmoid(gates[0]);
  auto forgetgate = torch::sigmoid(gates[1]);
  auto cellgate = torch::tanh(gates[2]);
  auto outgate = torch::sigmoid(gates[3]);

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * torch::tanh(cy);

  return LSTMState(hy, cy);
}

std::pair<torch::Tensor, LSTMState> lstm(const torch::Tensor& input, const LSTMState& start_state,
                                         const torch::Tensor& weight_ih, const torch::Tensor& weight_hh,
                                         const torch::Tensor& bias) {
  std::vector<torch::Tensor> outputs;
  outputs.reserve(input.size(0));
  auto state = start_state;
  for (auto& step : input.unbind(0)) {
    state = lstm_cell(step, state, weight_ih, weight_hh, bias);
    outputs.push_back(std::get<0>(state));
  }
  return std::make_pair(torch::stack(outputs), state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lstm_cell", lstm_cell);
  m.def("lstm", lstm);
}
