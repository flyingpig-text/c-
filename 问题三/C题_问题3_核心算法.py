# -*- coding: utf-8 -*-
"""
C 题 问题 3 核心算法模块（独立函数版）
=================================================================
从《C题_问题3_NSGA2_TOPSIS.py》中抽出的通用算法层：

    * NSGA-II：非支配排序、拥挤距离、锦标赛选择、SBX 交叉、
      多项式变异、精英保留
    * TOPSIS：z-score 标准化、minmax 方向处理、向量归一化、
      正负理想解、贴近度排序、权重敏感性

本模块中的函数不依赖问题常量，输入输出均为显式参数；
具体问题的材料表、海水温度预测、目标/约束计算通过 evaluate 闭包注入。
论文“算法设计”章节可直接引用本模块的函数签名与接口说明。

依赖：numpy / pandas
运行示例：python C题_问题3_核心算法.py
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# ==================================================================
# 1. 个体数据结构
# ==================================================================
class Individual:
    """NSGA-II 种群个体。

    输入：
        chrom    : 决策变量向量，一维数组
        obj      : 最小化目标向量，一维数组
        feasible : 是否满足全部约束
        viol     : 约束违和量，g>=0 约束下为 sum(max(0,-g))
        res      : 可选，保存该个体的详细物理量/结果字典
    """

    __slots__ = ("chrom", "obj", "feasible", "viol", "rank", "crowd", "res")

    def __init__(self, chrom, obj, feasible, viol, res=None):
        self.chrom = chrom
        self.obj = obj
        self.feasible = feasible
        self.viol = viol
        self.rank = 0
        self.crowd = 0.0
        self.res = res


# ==================================================================
# 2. 非支配排序
# ==================================================================
def dominates(p: Individual, q: Individual, tol: float = 1e-12) -> bool:
    """约束支配判断。

    输入：
        p, q : Individual
        tol  : 浮点比较容差
    输出：
        bool，p 支配 q 时返回 True。
    规则：
        1) 可行个体支配不可行个体；
        2) 两个不可行个体按违和量 viol 小者支配；
        3) 两个可行个体按帕累托支配：p 各目标不差于 q 且至少一项更优。
    """
    if p.feasible != q.feasible:
        return p.feasible and not q.feasible
    if not p.feasible:
        return p.viol < q.viol - tol
    return bool(np.all(p.obj <= q.obj + tol)
                and np.any(p.obj < q.obj - tol))


def fast_non_dominated_sort(pop: list) -> list[list[int]]:
    """快速非支配排序（NSGA-II 标准算法）。

    输入：
        pop : Individual 列表
    输出：
        list[list[int]]，每层为个体索引列表；
        第 0 层为当前种群的非支配前沿。
    复杂度：
        O(M * N^2)，M 为目标数，N 为种群规模。
    """
    n = len(pop)
    dominated = [set() for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(pop[i], pop[j]):
                dominated[i].add(j)
            elif dominates(pop[j], pop[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            pop[i].rank = 0
            fronts[0].append(i)

    k = 0
    while k < len(fronts) and fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)
        k += 1
        if nxt:
            fronts.append(nxt)
    return [f for f in fronts if f]


# ==================================================================
# 3. 拥挤距离
# ==================================================================
def crowding_distance(pop: list, front_idx: list[int]) -> None:
    """计算同一非支配层内个体的拥挤距离，原地写入 pop[i].crowd。

    输入：
        pop       : Individual 列表
        front_idx : 同一非支配层的个体索引列表
    输出：
        None；函数会修改每个个体的 crowd 属性。
    规则：
        边界个体拥挤距离设为 inf；
        内部个体按每个目标相邻个体的目标差/该目标极差累加。
    """
    m = len(pop[0].obj)
    front = [pop[i] for i in front_idx]
    for ind in front:
        ind.crowd = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowd = float("inf")
        return

    feasible_idx = [i for i, ind in enumerate(front) if ind.feasible]
    infeasible_idx = [i for i, ind in enumerate(front) if not ind.feasible]
    for i in feasible_idx:
        front[i].crowd = 0.0

    if feasible_idx:
        for mj in range(m):
            order = sorted(feasible_idx, key=lambda i: front[i].obj[mj])
            front[order[0]].crowd = float("inf")
            front[order[-1]].crowd = float("inf")
            rng = front[order[-1]].obj[mj] - front[order[0]].obj[mj]
            if rng < 1e-12:
                continue
            for i in range(1, len(order) - 1):
                if front[order[i]].crowd != float("inf"):
                    front[order[i]].crowd += (
                        front[order[i + 1]].obj[mj]
                        - front[order[i - 1]].obj[mj]) / rng

    if infeasible_idx:
        order = sorted(infeasible_idx, key=lambda i: front[i].viol)
        front[order[0]].crowd = float("inf")
        front[order[-1]].crowd = float("inf")
        rng = front[order[-1]].viol - front[order[0]].viol
        if rng > 1e-12:
            for i in range(1, len(order) - 1):
                if front[order[i]].crowd != float("inf"):
                    front[order[i]].crowd += (
                        front[order[i + 1]].viol
                        - front[order[i - 1]].viol) / rng


# ==================================================================
# 4. 遗传算子
# ==================================================================
def tournament_select(pop: list, rng: np.random.Generator,
                      k: int = 2) -> Individual:
    """二元锦标赛选择。

    输入：
        pop : Individual 列表
        rng : numpy 随机数生成器
        k   : 锦标赛个体数，默认 2
    输出：
        Individual，被选中的父代个体。
    规则：
        先比较 rank，rank 小者胜；rank 相同时 crowd 大者胜。
    """
    idx = rng.integers(0, len(pop), size=k)
    best = pop[idx[0]]
    for i in idx[1:]:
        other = pop[i]
        if other.rank < best.rank:
            best = other
        elif other.rank == best.rank and other.crowd > best.crowd:
            best = other
    return best


def sbx_crossover(p1: float, p2: float, lo: float, hi: float,
                  rng: np.random.Generator, pc: float = 0.90,
                  eta_c: float = 15.0) -> tuple[float, float]:
    """模拟二进制交叉 SBX，用于实数基因。

    输入：
        p1, p2 : 父代实数基因
        lo, hi : 基因下界、上界
        rng    : 随机数生成器
        pc     : 交叉概率
        eta_c  : SBX 分布指数，越大子代越接近父代
    输出：
        (child1, child2)，两个子代实数基因，且被裁剪到 [lo, hi]。
    """
    if rng.random() > pc:
        return float(p1), float(p2)
    u = rng.random()
    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
    else:
        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return float(np.clip(c1, lo, hi)), float(np.clip(c2, lo, hi))


def polynomial_mutation(x: float, lo: float, hi: float,
                        rng: np.random.Generator, pm: float = 0.10,
                        eta_m: float = 20.0) -> float:
    """多项式变异，用于实数基因。

    输入：
        x     : 待变异基因
        lo,hi : 基因边界
        rng   : 随机数生成器
        pm    : 变异概率
        eta_m : 变异分布指数
    输出：
        float，变异后的基因，裁剪到 [lo, hi]。
    """
    if rng.random() > pm:
        return float(x)
    u = rng.random()
    if u < 0.5:
        delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
    else:
        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
    return float(np.clip(x + delta * (hi - lo), lo, hi))


# ==================================================================
# 5. NSGA-II 主循环
# ==================================================================
def default_initializer(evaluate: Callable, bounds: Sequence[tuple[float, float]],
                        pop_size: int, rng: np.random.Generator) -> list:
    """默认随机初始化：所有基因在边界内均匀采样。

    输入：
        evaluate : Callable，输入 chrom 向量，返回 Individual
        bounds   : [(lo, hi), ...] 每维决策变量边界
        pop_size : 种群规模
        rng      : 随机数生成器
    输出：
        list[Individual]，初始种群。
    """
    pop = []
    for _ in range(pop_size):
        chrom = [rng.uniform(lo, hi) for lo, hi in bounds]
        pop.append(evaluate(chrom))
    return pop


def nsga2(
    evaluate: Callable,
    bounds: Sequence[tuple[float, float]],
    pop_size: int = 80,
    generations: int = 120,
    seed: int | None = None,
    pc: float = 0.90,
    pm: float = 0.10,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    discrete_indices: Sequence[int] = (),
    initializer: Callable | None = None,
    verbose_every: int = 20,
) -> tuple[list, list, list]:
    """NSGA-II 主循环。

    输入：
        evaluate     : Callable，输入 chrom 返回 Individual。
                      具体问题通过该闭包注入目标、约束和详细结果。
        bounds       : [(lo, hi), ...]，决策变量边界。
        pop_size     : 种群规模。
        generations  : 迭代代数。
        seed         : 随机种子；None 表示不固定。
        pc, pm       : 交叉概率、变异概率。
        eta_c, eta_m : SBX 和多项式变异分布指数。
        discrete_indices : 整数/离散基因索引；这些基因使用均匀交叉，
                           变异时在整数边界内随机取值。
        initializer  : 可选，自定义初始种群函数
                       signature: (evaluate, bounds, pop_size, rng) -> pop
        verbose_every: 每隔多少代打印一次收敛信息。
    输出：
        (pop, pareto, history)
            pop     : 最终种群 list[Individual]
            pareto  : 第 0 层可行个体的 res 字典列表
            history : [(gen, front_size, best_Q, mean_cost, crowd), ...]
    """
    rng = np.random.default_rng(seed)
    if initializer is None:
        initializer = default_initializer
    pop = initializer(evaluate, bounds, pop_size, rng)
    history = []

    for gen in range(1, generations + 1):
        fronts = fast_non_dominated_sort(pop)
        for f_idx in fronts:
            crowding_distance(pop, f_idx)

        parents = [tournament_select(pop, rng) for _ in range(pop_size)]
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < pop_size else parents[0]
            child = []
            for j, (lo, hi) in enumerate(bounds):
                if j in discrete_indices:
                    # 离散基因：均匀交叉 + 整数边界随机变异
                    c1 = p1.chrom[j] if rng.random() < 0.5 else p2.chrom[j]
                    c2 = p2.chrom[j] if rng.random() < 0.5 else p1.chrom[j]
                    if rng.random() < pm:
                        c1 = float(rng.integers(int(lo), int(hi) + 1))
                    if rng.random() < pm:
                        c2 = float(rng.integers(int(lo), int(hi) + 1))
                else:
                    c1, c2 = sbx_crossover(p1.chrom[j], p2.chrom[j],
                                           lo, hi, rng, pc, eta_c)
                    c1 = polynomial_mutation(c1, lo, hi, rng, pm, eta_m)
                    c2 = polynomial_mutation(c2, lo, hi, rng, pm, eta_m)
                child.append([c1, c2])
            c1v = [pair[0] for pair in child]
            c2v = [pair[1] for pair in child]
            offspring.append(evaluate(c1v))
            offspring.append(evaluate(c2v))
        offspring = offspring[:pop_size]

        combined = pop + offspring
        fronts = fast_non_dominated_sort(combined)
        for f_idx in fronts:
            crowding_distance(combined, f_idx)

        new_pop = []
        for f_idx in fronts:
            if len(new_pop) + len(f_idx) <= pop_size:
                new_pop.extend(combined[i] for i in f_idx)
            else:
                need = pop_size - len(new_pop)
                if need <= 0:
                    break
                order = sorted(f_idx, key=lambda i: -combined[i].crowd)
                new_pop.extend(combined[i] for i in order[:need])
                break
        pop = new_pop

        fea = [ind for ind in pop if ind.feasible]
        if fea:
            q_vals = [ind.res["Q"] for ind in fea
                      if ind.res and "Q" in ind.res]
            cost_vals = [ind.res["cost"] for ind in fea
                         if ind.res and "cost" in ind.res]
            best_q = max(q_vals) if q_vals else 0.0
            mean_cost = float(np.mean(cost_vals)) if cost_vals else 0.0
            front_sz = len([ind for ind in pop
                            if ind.feasible and ind.rank == 0])
            crowd = float(np.mean([ind.crowd for ind in fea
                                   if np.isfinite(ind.crowd)]) or 0.0)
        else:
            best_q, mean_cost, front_sz, crowd = 0.0, 0.0, 0, 0.0
        history.append((gen, front_sz, best_q, mean_cost, crowd))
        if gen % verbose_every == 0 or gen == generations:
            print("    第 %3d 代：前沿可行个体 %3d，最好 Q=%10.1f W，"
                  "前沿平均成本=%10.0f 元" % (gen, front_sz, best_q, mean_cost))

    fronts = fast_non_dominated_sort(pop)
    front0 = [pop[i] for i in fronts[0] if pop[i].feasible]
    pareto = [ind.res for ind in front0 if ind.res is not None]
    return pop, pareto, history


# ==================================================================
# 6. TOPSIS 决策
# ==================================================================
def zscore_matrix(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """z-score 标准化。

    输入：
        df   : 方案矩阵
        cols : 需要标准化的列名
    输出：
        DataFrame，各列均值为 0、标准差为 1。
    """
    return (df[list(cols)] - df[list(cols)].mean()) / df[list(cols)].std()


def minmax_directed(z: pd.DataFrame,
                    cost_cols: Sequence[str] = ()) -> pd.DataFrame:
    """极差变换并统一方向。

    输入：
        z         : z-score 后的矩阵
        cost_cols : 成本型列名，越小越好
    输出：
        DataFrame，成本型列取反变换，其余列按效益型变换，
        所有值位于 [0, 1]。
    """
    cost_set = set(cost_cols)
    out = pd.DataFrame(index=z.index)
    for col in z.columns:
        mn, mx = z[col].min(), z[col].max()
        if mx - mn < 1e-12:
            out[col] = 1.0
        elif col in cost_set:
            out[col] = (mx - z[col]) / (mx - mn)
        else:
            out[col] = (z[col] - mn) / (mx - mn)
    return out


def topsis(
    pareto_df: pd.DataFrame,
    weights: np.ndarray,
    benefit_cols: Sequence[str] | None = None,
    cost_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """TOPSIS 完整决策流程。

    输入：
        pareto_df   : 帕累托方案表
        weights     : 权重向量，长度等于参与列数
        benefit_cols: 效益型列名，越大越好；默认取除 cost_cols 外的数值列
        cost_cols   : 成本型列名，越小越好
    输出：
        DataFrame，在原表基础上追加
        D_plus、D_minus、TOPSIS贴近度、排名，并按贴近度降序。
    """
    if cost_cols is None:
        cost_cols = ()
    if benefit_cols is None:
        benefit_cols = [c for c in pareto_df.columns
                        if c not in cost_cols
                        and pd.api.types.is_numeric_dtype(pareto_df[c])]
    cols = list(benefit_cols) + list(cost_cols)
    if len(weights) != len(cols):
        raise ValueError("权重长度必须等于 benefit_cols + cost_cols 数量")

    z = zscore_matrix(pareto_df, cols)
    mm = minmax_directed(z, cost_cols)
    norm = mm / np.sqrt((mm ** 2).sum(axis=0))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    v = norm.mul(w, axis=1)

    v_pos = v.max(axis=0)
    v_neg = v.min(axis=0)
    d_pos = np.sqrt(((v - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - v_neg) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg)

    result = pareto_df.copy()
    result["D_plus"] = d_pos.to_numpy()
    result["D_minus"] = d_neg.to_numpy()
    result["TOPSIS贴近度"] = closeness.to_numpy()
    result = result.sort_values("TOPSIS贴近度", ascending=False).reset_index(drop=True)
    result.insert(0, "排名", np.arange(1, len(result) + 1))
    return result


# ==================================================================
# 7. 权重敏感性分析
# ==================================================================
def spearman_rank(x, y) -> float:
    """Spearman 秩相关系数，不依赖 scipy。"""
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def weight_sensitivity(
    pareto_df: pd.DataFrame,
    weights: np.ndarray,
    benefit_cols: Sequence[str],
    cost_cols: Sequence[str],
    deltas: Sequence[float] = (0.05, 0.10),
) -> pd.DataFrame:
    """TOPSIS 权重 ±delta 扰动敏感性。

    输入：
        pareto_df   : 帕累托方案表
        weights     : 基准权重
        benefit_cols: 效益型列名
        cost_cols   : 成本型列名
        deltas      : 扰动幅度序列
    输出：
        DataFrame，包含扰动项、扰动幅度、扰动后权重、Spearman 秩相关、
        排名变化数。
    """
    base = topsis(pareto_df, weights, benefit_cols, cost_cols)
    base_rank = base["排名"].to_numpy()
    names = list(benefit_cols) + list(cost_cols)
    rows = []
    for i, name in enumerate(names):
        for delta in deltas:
            for sign in (1, -1):
                w = np.asarray(weights, dtype=float).copy()
                w[i] += sign * delta
                w = w / w.sum()
                tmp = topsis(pareto_df, w, benefit_cols, cost_cols)
                rho = spearman_rank(base_rank, tmp["排名"].to_numpy())
                rows.append({
                    "扰动项": name,
                    "扰动": "%+.0f%%" % (sign * delta * 100),
                    "权重": "%.2f/%.2f/%.2f" % tuple(w),
                    "Spearman": rho,
                    "排名变化数": int(np.sum(base_rank != tmp["排名"].to_numpy())),
                })
    return pd.DataFrame(rows)


# ==================================================================
# 8. 简单基准算例
# ==================================================================
def run_demo() -> None:
    """用不含问题常量的小算例验证 NSGA-II + TOPSIS 可独立运行。"""
    print("=" * 70)
    print("核心算法基准算例")
    print("目标 1：最大化 f1 = x")
    print("目标 2：最小化 f2 = (x - 0.5)^2")
    print("=" * 70)

    def evaluate(chrom):
        x = float(chrom[0])
        f1 = x
        f2 = (x - 0.5) ** 2
        obj = np.array([-f1, f2])   # 最小化形式
        res = {"x": x, "f1_benefit": f1, "f2_cost": f2,
               "Q": f1, "cost": f2, "obj": obj.copy()}
        return Individual(chrom.copy(), obj, True, 0.0, res)

    pop, pareto, history = nsga2(
        evaluate=evaluate,
        bounds=[(0.0, 1.0)],
        pop_size=40,
        generations=30,
        seed=1,
        verbose_every=10,
    )
    df = pd.DataFrame(pareto)
    result = topsis(df, np.array([0.5, 0.5]),
                    benefit_cols=["f1_benefit"],
                    cost_cols=["f2_cost"])
    print("\n帕累托前沿方案数：", len(df))
    print("TOPSIS 前 5 名：")
    print(result.head(5).to_string(index=False))


if __name__ == "__main__":
    run_demo()
