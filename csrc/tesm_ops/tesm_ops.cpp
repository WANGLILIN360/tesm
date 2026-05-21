#include <torch/extension.h>

at::Tensor chunk_state_scan_fwd_cuda(const at::Tensor& decay, const at::Tensor& update, int64_t chunk_size);
std::vector<at::Tensor> chunk_state_scan_bwd_cuda(const at::Tensor& decay, const at::Tensor& states, const at::Tensor& grad_states);
at::Tensor local_entanglement_fwd_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& values, const at::Tensor& local_bias, double threshold);
std::vector<at::Tensor> local_entanglement_bwd_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& values, const at::Tensor& local_bias, const at::Tensor& grad_out, double threshold);
at::Tensor quantized_linear_fwd_cuda(const at::Tensor& x, const at::Tensor& qweight);
at::Tensor quantized_linear_fwd_bias_cuda(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& bias);
std::vector<at::Tensor> quantized_linear_bwd_cuda(const at::Tensor& grad_output, const at::Tensor& x, const at::Tensor& qweight, bool has_bias);
at::Tensor int2_linear_fwd_cuda(const at::Tensor& x, const at::Tensor& packed_weight, const at::Tensor& scale, const at::Tensor* bias);
at::Tensor int2_linear_optimized_fwd_cuda(const at::Tensor& x, const at::Tensor& packed_weight, const at::Tensor& scale, const at::Tensor* bias);
at::Tensor int8xint2_linear_fwd_cuda(const at::Tensor& x_quant, const at::Tensor& x_scale, const at::Tensor& packed_weight, const at::Tensor& weight_scale, const at::Tensor* bias);

// Global Entanglement
at::Tensor tesm_global_entanglement_fwd_cuda(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold);
at::Tensor tesm_global_entanglement_bwd_cuda(const at::Tensor& grad_out, const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold);
at::Tensor tesm_global_entanglement_mimo_fwd_cuda(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold);
at::Tensor tesm_global_entanglement_mimo_bwd_cuda(const at::Tensor& grad_out, const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold);

// Fused Output
at::Tensor tesm_fused_output_fwd_cuda(const at::Tensor& local, const at::Tensor& gate, const at::Tensor& state_proj, const at::Tensor& ent_proj, double ent_scale);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tesm_fused_output_bwd_cuda(const at::Tensor& grad_out, const at::Tensor& local, const at::Tensor& gate, double ent_scale);
at::Tensor tesm_fused_output_mimo_fwd_cuda(const at::Tensor& local, const at::Tensor& gate, const at::Tensor& state_proj, const at::Tensor& ent_proj, double ent_scale);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tesm_fused_output_mimo_bwd_cuda(const at::Tensor& grad_out, const at::Tensor& local, const at::Tensor& gate, double ent_scale);

// MIMO State Scan
at::Tensor chunk_state_scan_mimo_fwd_cuda(const at::Tensor& decay, const at::Tensor& update, int64_t chunk_size);
std::vector<at::Tensor> chunk_state_scan_mimo_bwd_cuda(const at::Tensor& decay, const at::Tensor& states, const at::Tensor& grad_states);

// MIMO Local Entanglement
at::Tensor local_entanglement_mimo_fwd_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& bias, double threshold);
std::vector<at::Tensor> local_entanglement_mimo_bwd_cuda(const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& bias, double threshold);

at::Tensor chunk_state_scan_fwd(const at::Tensor& decay, const at::Tensor& update, int64_t chunk_size) {
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(update.is_cuda(), "update must be CUDA");
    return chunk_state_scan_fwd_cuda(decay, update, chunk_size);
}

std::vector<at::Tensor> chunk_state_scan_bwd(const at::Tensor& decay, const at::Tensor& states, const at::Tensor& grad_states) {
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(states.is_cuda(), "states must be CUDA");
    TORCH_CHECK(grad_states.is_cuda(), "grad_states must be CUDA");
    return chunk_state_scan_bwd_cuda(decay, states, grad_states);
}

at::Tensor local_entanglement_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& values, const at::Tensor& local_bias, double threshold) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(local_bias.is_cuda(), "local_bias must be CUDA");
    return local_entanglement_fwd_cuda(q, k, values, local_bias, threshold);
}

std::vector<at::Tensor> local_entanglement_bwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& values, const at::Tensor& local_bias, const at::Tensor& grad_out, double threshold) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(local_bias.is_cuda(), "local_bias must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    return local_entanglement_bwd_cuda(q, k, values, local_bias, grad_out, threshold);
}

at::Tensor quantized_linear_fwd(const at::Tensor& x, const at::Tensor& qweight) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    return quantized_linear_fwd_cuda(x, qweight);
}

at::Tensor quantized_linear_fwd_bias(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    return quantized_linear_fwd_bias_cuda(x, qweight, bias);
}

std::vector<at::Tensor> quantized_linear_bwd(const at::Tensor& grad_output, const at::Tensor& x, const at::Tensor& qweight, bool has_bias) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    return quantized_linear_bwd_cuda(grad_output, x, qweight, has_bias);
}

at::Tensor int2_linear_fwd(const at::Tensor& x, const at::Tensor& packed_weight, const at::Tensor& scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    return int2_linear_fwd_cuda(x, packed_weight, scale, nullptr);
}

at::Tensor int2_linear_fwd_bias(const at::Tensor& x, const at::Tensor& packed_weight, const at::Tensor& scale, const at::Tensor& bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    return int2_linear_fwd_cuda(x, packed_weight, scale, &bias);
}

at::Tensor int2_linear_optimized_fwd(const at::Tensor& x, const at::Tensor& packed_weight, const at::Tensor& scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    return int2_linear_optimized_fwd_cuda(x, packed_weight, scale, nullptr);
}

at::Tensor int2_linear_optimized_fwd_bias(const at::Tensor& x, const at::Tensor& packed_weight, const at::Tensor& scale, const at::Tensor& bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    return int2_linear_optimized_fwd_cuda(x, packed_weight, scale, &bias);
}

at::Tensor int8xint2_linear_fwd(const at::Tensor& x_quant, const at::Tensor& x_scale, const at::Tensor& packed_weight, const at::Tensor& weight_scale) {
    TORCH_CHECK(x_quant.is_cuda(), "x_quant must be CUDA");
    TORCH_CHECK(x_scale.is_cuda(), "x_scale must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
    return int8xint2_linear_fwd_cuda(x_quant, x_scale, packed_weight, weight_scale, nullptr);
}

at::Tensor int8xint2_linear_fwd_bias(const at::Tensor& x_quant, const at::Tensor& x_scale, const at::Tensor& packed_weight, const at::Tensor& weight_scale, const at::Tensor& bias) {
    TORCH_CHECK(x_quant.is_cuda(), "x_quant must be CUDA");
    TORCH_CHECK(x_scale.is_cuda(), "x_scale must be CUDA");
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    return int8xint2_linear_fwd_cuda(x_quant, x_scale, packed_weight, weight_scale, &bias);
}

// Global Entanglement SISO
at::Tensor global_entanglement_fwd(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Bias.is_cuda(), "Bias must be CUDA");
    return tesm_global_entanglement_fwd_cuda(Q, K, V, Bias, threshold);
}

at::Tensor global_entanglement_bwd(const at::Tensor& grad_out, const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Bias.is_cuda(), "Bias must be CUDA");
    return tesm_global_entanglement_bwd_cuda(grad_out, Q, K, V, Bias, threshold);
}

// Global Entanglement MIMO
at::Tensor global_entanglement_mimo_fwd(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Bias.is_cuda(), "Bias must be CUDA");
    return tesm_global_entanglement_mimo_fwd_cuda(Q, K, V, Bias, threshold);
}

at::Tensor global_entanglement_mimo_bwd(const at::Tensor& grad_out, const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V, const at::Tensor& Bias, double threshold) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Bias.is_cuda(), "Bias must be CUDA");
    return tesm_global_entanglement_mimo_bwd_cuda(grad_out, Q, K, V, Bias, threshold);
}

// Fused Output SISO
at::Tensor fused_output_fwd(const at::Tensor& local, const at::Tensor& gate, const at::Tensor& state_proj, const at::Tensor& ent_proj, double ent_scale) {
    TORCH_CHECK(local.is_cuda(), "local must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(state_proj.is_cuda(), "state_proj must be CUDA");
    TORCH_CHECK(ent_proj.is_cuda(), "ent_proj must be CUDA");
    return tesm_fused_output_fwd_cuda(local, gate, state_proj, ent_proj, ent_scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_output_bwd(const at::Tensor& grad_out, const at::Tensor& local, const at::Tensor& gate, double ent_scale) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(local.is_cuda(), "local must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    return tesm_fused_output_bwd_cuda(grad_out, local, gate, ent_scale);
}

// Fused Output MIMO
at::Tensor fused_output_mimo_fwd(const at::Tensor& local, const at::Tensor& gate, const at::Tensor& state_proj, const at::Tensor& ent_proj, double ent_scale) {
    TORCH_CHECK(local.is_cuda(), "local must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(state_proj.is_cuda(), "state_proj must be CUDA");
    TORCH_CHECK(ent_proj.is_cuda(), "ent_proj must be CUDA");
    return tesm_fused_output_mimo_fwd_cuda(local, gate, state_proj, ent_proj, ent_scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_output_mimo_bwd(const at::Tensor& grad_out, const at::Tensor& local, const at::Tensor& gate, double ent_scale) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(local.is_cuda(), "local must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    return tesm_fused_output_mimo_bwd_cuda(grad_out, local, gate, ent_scale);
}

// MIMO State Scan
at::Tensor chunk_state_scan_mimo_fwd(const at::Tensor& decay, const at::Tensor& update, int64_t chunk_size) {
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(update.is_cuda(), "update must be CUDA");
    return chunk_state_scan_mimo_fwd_cuda(decay, update, chunk_size);
}

std::vector<at::Tensor> chunk_state_scan_mimo_bwd(const at::Tensor& decay, const at::Tensor& states, const at::Tensor& grad_states) {
    TORCH_CHECK(decay.is_cuda(), "decay must be CUDA");
    TORCH_CHECK(states.is_cuda(), "states must be CUDA");
    TORCH_CHECK(grad_states.is_cuda(), "grad_states must be CUDA");
    return chunk_state_scan_mimo_bwd_cuda(decay, states, grad_states);
}

// MIMO Local Entanglement
at::Tensor local_entanglement_mimo_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& bias, double threshold) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    return local_entanglement_mimo_fwd_cuda(q, k, v, bias, threshold);
}

std::vector<at::Tensor> local_entanglement_mimo_bwd(const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& bias, double threshold) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    return local_entanglement_mimo_bwd_cuda(grad_out, q, k, v, bias, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_state_scan_fwd", &chunk_state_scan_fwd, "TESM chunk state scan forward");
    m.def("chunk_state_scan_bwd", &chunk_state_scan_bwd, "TESM chunk state scan backward");
    m.def("local_entanglement_fwd", &local_entanglement_fwd, "TESM local entanglement forward");
    m.def("local_entanglement_bwd", &local_entanglement_bwd, "TESM local entanglement backward");
    m.def("quantized_linear_fwd", &quantized_linear_fwd, "TESM quantized linear forward");
    m.def("quantized_linear_fwd_bias", &quantized_linear_fwd_bias, "TESM quantized linear forward with bias");
    m.def("quantized_linear_bwd", &quantized_linear_bwd, "TESM quantized linear backward");
    m.def("int2_linear_fwd", &int2_linear_fwd, "TESM INT2 linear forward");
    m.def("int2_linear_fwd_bias", &int2_linear_fwd_bias, "TESM INT2 linear forward with bias");
    m.def("int2_linear_optimized_fwd", &int2_linear_optimized_fwd, "TESM INT2 linear optimized forward");
    m.def("int2_linear_optimized_fwd_bias", &int2_linear_optimized_fwd_bias, "TESM INT2 linear optimized forward with bias");
    m.def("int8xint2_linear_fwd", &int8xint2_linear_fwd, "TESM INT8xINT2 linear forward");
    m.def("int8xint2_linear_fwd_bias", &int8xint2_linear_fwd_bias, "TESM INT8xINT2 linear forward with bias");
    // Global Entanglement
    m.def("global_entanglement_fwd", &global_entanglement_fwd, "TESM global entanglement forward (SISO)");
    m.def("global_entanglement_bwd", &global_entanglement_bwd, "TESM global entanglement backward (SISO)");
    m.def("global_entanglement_mimo_fwd", &global_entanglement_mimo_fwd, "TESM global entanglement forward (MIMO)");
    m.def("global_entanglement_mimo_bwd", &global_entanglement_mimo_bwd, "TESM global entanglement backward (MIMO)");
    // Fused Output
    m.def("fused_output_fwd", &fused_output_fwd, "TESM fused output forward (SISO)");
    m.def("fused_output_bwd", &fused_output_bwd, "TESM fused output backward (SISO)");
    m.def("fused_output_mimo_fwd", &fused_output_mimo_fwd, "TESM fused output forward (MIMO)");
    m.def("fused_output_mimo_bwd", &fused_output_mimo_bwd, "TESM fused output backward (MIMO)");
    // MIMO State Scan
    m.def("chunk_state_scan_mimo_fwd", &chunk_state_scan_mimo_fwd, "TESM chunk state scan forward (MIMO)");
    m.def("chunk_state_scan_mimo_bwd", &chunk_state_scan_mimo_bwd, "TESM chunk state scan backward (MIMO)");
    // MIMO Local Entanglement
    m.def("local_entanglement_mimo_fwd", &local_entanglement_mimo_fwd, "TESM local entanglement forward (MIMO)");
    m.def("local_entanglement_mimo_bwd", &local_entanglement_mimo_bwd, "TESM local entanglement backward (MIMO)");
}
