import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    total_norms = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # === 计算梯度范数（L2） ===
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        total_norms.append(total_norm)

        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tGrad Norm: {total_norm:.4f}")

    return total_norms  # 可选：返回整轮梯度范数序列
