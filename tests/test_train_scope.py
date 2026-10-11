"""Head adaptation cannot silently update the visual encoder or BN buffers."""
import torch

from model import TinyVNOCR
from restore_blob import freeze_bn_stats
from train2 import train_scope


def test_head_step_preserves_every_encoder_parameter_and_buffer():
    torch.set_num_threads(2)
    torch.manual_seed(111)
    model = TinyVNOCR(5, width=1.0, bottleneck=96, depth=1, ctx=2, ctx_k=5)
    train_scope(model, "head")
    model.train()
    freeze_bn_stats(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(torch.randn(2, 1, 32, 64)).log_softmax(-1)
    loss = torch.nn.functional.ctc_loss(logits, torch.tensor([1, 2, 2, 3]),
        torch.tensor([8, 8]), torch.tensor([2, 2]), blank=0)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith("head."))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("head."))
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("head."))
