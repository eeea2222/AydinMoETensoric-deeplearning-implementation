import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time

# --- 1. AYDIN CONFIG ---
@dataclass
class AydinConfig:
    hidden_dim: int = 512
    intermediate_dim: int = 1024
    num_experts: int = 8      # Toplam uzman sayısı
    top_k: int = 2            # Her token için aktifleşecek uzman sayısı
    dropout: float = 0.1

# --- 2. TENSORIC ENGINE: The Vectorized Expert Layer ---
class TensoricSwiGLU(nn.Module):
    """
    Python döngüleri yerine Tensor indexing kullanarak çalışan
    vektörize edilmiş MoE katmanı.
    """
    def __init__(self, config: AydinConfig):
        super().__init__()
        self.config = config
        
        # Tüm uzmanların ağırlıkları devasa tensörlerde tutulur.
        # w13: Gate ve Up projeksiyonları birleşik (Fused)
        self.w13 = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.intermediate_dim * 2))
        
        # w2: Down projeksiyonu
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_dim, config.hidden_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Ağırlıkları güvenli başlat (std=0.02)
        nn.init.normal_(self.w13, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Seq, Hidden)
        expert_indices: (Batch, Seq, TopK)
        """
        B, S, H = x.shape
        K = self.config.top_k
        
        # 1. Ağırlık Seçimi (Weight Gathering)
        # UYARI: VRAM kullanımı yüksektir. 
        # (B, S, K, H, 2*I) boyutunda tensör oluşturur.
        w13_selected = self.w13[expert_indices] 
        
        # 2. Vektörize Matris Çarpımı
        x_expanded = x.view(B, S, 1, 1, H)
        h13 = torch.matmul(x_expanded, w13_selected)
        
        # 3. SwiGLU Aktivasyonu
        gate, up = h13.split(self.config.intermediate_dim, dim=-1)
        h_inter = F.silu(gate) * up 
        
        # 4. Down Projection
        w2_selected = self.w2[expert_indices] 
        out = torch.matmul(h_inter, w2_selected) 
        
        return out.squeeze(-2)

# --- 3. ARCHITECT: AydinMoE "Tensoric" Edition ---
class AydinMoETensoric(nn.Module):
    def __init__(self, config: AydinConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.experts = TensoricSwiGLU(config)

    def forward(self, x: torch.Tensor):
        # --- 1. ROUTING ---
        router_logits = self.router(x)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K
        weights, indices = torch.topk(routing_probs, self.config.top_k, dim=-1)
        
        # Normalize weights (Sıfıra bölme hatasını önlemek için +1e-6 ekledik)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # --- 2. TENSORIC COMPUTATION ---
        expert_output = self.experts(x, indices)
        
        # --- 3. WEIGHTED COMBINATION ---
        weighted_output = expert_output * weights.unsqueeze(-1)
        final_output = weighted_output.sum(dim=2)
        
        return final_output

# --- 4. THE COMPILER SETUP ---
def get_compiled_model():
    if not torch.cuda.is_available():
        raise RuntimeError("Bu kod NVIDIA GPU gerektirir!")

    conf = AydinConfig()
    # Modeli GPU'ya at ve bfloat16 yap
    model = AydinMoETensoric(conf).cuda().to(dtype=torch.bfloat16)
    
    torch.set_float32_matmul_precision('high')
    
    print("🚀 Model derleniyor (torch.compile)...")
    compiled_model = torch.compile(model, mode="max-autotune")
    
    return compiled_model

# --- 5. BENCHMARK ---
if __name__ == "__main__":
    try:
        model = get_compiled_model()
        
        # Bellek hatası almamak için Batch Size 16 seçildi.
        # RTX 3090/4090 varsa 32 yapabilirsin.
        BATCH_SIZE = 16 
        SEQ_LEN = 128
        HIDDEN_DIM = 512
        
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
        
        print(f"📊 Veri Boyutu: [{BATCH_SIZE}, {SEQ_LEN}, {HIDDEN_DIM}]")
        
        # Warmup
        print("🔥 Isınıyor...")
        start = time.time()
        for _ in range(3):
            _ = model(x)
        print(f"✅ Isınma Bitti: {time.time() - start:.2f} sn")

        # Benchmark
        ITERATIONS = 1000
        print(f"\n⚡ BENCHMARK ({ITERATIONS} iterasyon) ⚡")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(ITERATIONS):
            _ = model(x)
            
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"⏱️  Toplam Süre: {elapsed_ms:.2f} ms")
        print(f"🚀 İterasyon Başına: {elapsed_ms/ITERATIONS:.3f} ms")
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("⚠️  'Out of memory' hatası ise BATCH_SIZE değişkenini küçültün (örn: 8).")
