# AydinMoETensoric-deeplearning-implementation
AydinMoE: High-Performance Vectorized Sparse Mixture-of-Experts
## References & Acknowledgements

This implementation is based on the following foundational research and modern architectures:

**1. The Original MoE Architecture:**
*   **Paper:** *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"*
*   **Authors:** Noam Shazeer et al. (Google Brain)
*   **Significance:** Defines the core logic of Top-K gating and sparse expert routing used in this module.

**2. Activation Function (SwiGLU):**
*   **Paper:** *"GLU Variants Improve Transformer"*
*   **Author:** Noam Shazeer (Google)
*   **Significance:** Source of the Swish-Gated Linear Unit (SwiGLU) activation mechanism implemented here for superior performance over ReLU/GELU.

**3. Architectural Configuration:**
*   **Model:** **Mixtral 8x7B** (Mistral AI)
*   **Paper:** *"Mixtral of Experts"*
*   **Significance:** This implementation adopts the effective `num_experts=8` and `top_k=2` configuration popularized by the Mixtral architecture.

**4. Framework & Optimization:**
*   **Library:** **PyTorch 2.0+** (Meta AI)
*   **Technology:** `torch.compile` & `torch.func`
*   **Significance:** Leveraging the latest compiler stack for kernel fusion and vectorized execution without custom CUDA kernels.

Bu modül, modern Büyük Dil Modelleri (LLM) için geliştirilmiş, vektörize edilmiş (tensoric) bir Mixture-of-Experts (MoE) mimarisidir. Geleneksel döngüsel (loop-based) uzman çağrıları yerine, tüm uzmanları tek bir devasa tensörde toplayarak GPU'nun paralel işlem kapasitesini %100 kullanır.

Teknik Özellikler (The Secret Sauce):

Tamamen Vektörize İşlem: Python döngüleri (for loops) ve CUDA stream yönetimi kaldırıldı. Bunun yerine gelişmiş gather ve matmul işlemleriyle tüm uzmanlar aynı anda hesaplanır.
Static Graph & Compilation Ready: Dinamik akış içermediği için torch.compile (PyTorch 2.0+) ile mükemmel uyumludur. Kod, tek bir "Fused Kernel" olarak derlenir.
Fused SwiGLU: Aktivasyon fonksiyonu, LLaMA ve PaLM mimarilerindeki gibi Gated Linear Unit (GLU) yapısındadır.
Top-K Routing: Her token için dinamik olarak en iyi $K$ uzmanı seçen, türevlenebilir (differentiable) bir yönlendirici (router) kullanır.
