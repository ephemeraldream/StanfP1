import numpy as np
import torch

def data_loader(
    x: np.ndarray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(x, dtype=torch.long)

    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {tuple(x.shape)}")

    n = x.size(0)
    if n <= context_length:
        raise ValueError(
            f"x is too short: len(x)={n}, context_length={context_length}"
        )

    starts = torch.randint(
        low=0,
        high=n - context_length,
        size=(batch_size,)
    )

    #  [0, 1, 2, ..., context_length - 1]
    offsets = torch.arange(context_length)

    # 3) строим матрицу индексов размера (batch_size, context_length)
    idx = starts[:, None] + offsets[None, :]

    # 4) сразу достаём все окна
    data = x[idx]
    targets = x[idx + 1]

    return data.to(device), targets.to(device)


# Ниже тестовый пример:
if __name__ == "__main__":
    # Test array из случайных целых чисел от 0 до 99, размером 20 элементов
    rng = np.random.default_rng(seed=42)
    test_np = rng.integers(low=0, high=100, size=20)
    batch_size = 4
    context_length = 5
    device = "cpu"

    data, targets = data_loader(
        test_np,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )
    # Точки останова удобно ставить здесь:
    print("Исходный массив:", test_np)
    print("data:\n", data)
    print("targets:\n", targets)