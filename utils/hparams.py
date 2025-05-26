from types import SimpleNamespace


def get_hyperparams(optimizer='Adam', lr=0.001):
    """
    返回超参数对象，支持传入优化器名称和学习率
    """
    return SimpleNamespace(
        batch_size=64,
        test_batch_size=1000,
        epochs=15,
        learning_rate=lr,
        optimizer=optimizer,
        seed=42,
        use_cuda=True,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        log_interval=100
    )