# -*- coding: utf-8 -*-
"""轻量字符错误率（CER）实现。"""


def _edit_distance(ref: list, hyp: list) -> tuple:
    """返回 (S, D, I, C) — 替换、删除、插入、正确数。"""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # deletion
                    dp[i][j - 1],    # insertion
                    dp[i - 1][j - 1] # substitution
                )

    # 回溯统计 S/D/I/C
    i, j = m, n
    S = D = I = C = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            C += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I, C


def compute_cer(predictions, references) -> float:
    """直接计算全局字符错误率，训练代码无需使用已弃用的 load_metric。"""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions/references 数量不一致: "
            f"{len(predictions)} != {len(references)}"
        )
    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        S, D, I, C = _edit_distance(list(reference), list(prediction))
        incorrect += S + D + I
        total += S + D + C
    return incorrect / total if total > 0 else 0.0
