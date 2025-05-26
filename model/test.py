import torch
import torch.nn.functional as F

def test(model, device, test_loader):
    """
    测试模型性能（返回平均 loss 和 accuracy）
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 测试时不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加loss
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)\n")
    return test_loss, accuracy