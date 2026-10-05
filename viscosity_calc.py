# -*- coding: utf-8 -*-
"""
viscosity_calc.py — Iida模型黏度计算（最终版）

基础：Forsbacka 1600°C (a=1.056, b=0.0058, c=-1.115)
标定参数（基于83条高碱度钢铁渣，Bi*≥1.0，MAPE=29.3%）：
  A_scale=1.60, E_scale=0.65, μ₀_scale=1.40
  α_Al2O3=0.02, α_SiO2=1.70, α_FeO=1.40

包含：
  1. Iida 机理模型：calc_iida_viscosity
  2. 光学碱度：calc_optical_basicity
  3. Gradient Boosting 数据驱动模型：calc_viscosity_ml
     （需同目录下 gb_viscosity_model.pkl）
  4. gb_model_available()：供 UI 判断 pkl 是否存在

修订记录：
  - _mass_to_mole 增加未知组分过滤，避免 KeyError
  - _load_gb_model 增加文件存在性检查与明确报错
  - 新增 gb_model_available()
  - 移除 warnings.warn，统一走返回值 warnings_list
"""

import math
import os
import pickle

import numpy as np

R = 8.314

# =====================================================================
# 组分基础参数表
# =====================================================================
COMPONENT_PARAMS = {
    "SiO2":  {"Tm": 2001, "M": 60.08,  "V": 27.29, "alpha": 1.70, "type": "A",
              "mu0": [(1573, 0.00467), (1673, 0.00378), (1773, 0.00313)]},
    "Al2O3": {"Tm": 2327, "M": 101.96, "V": 38.00, "alpha": 0.02, "type": "AM",
              "mu0": [(1573, 0.01022), (1673, 0.00795), (1773, 0.00635)]},
    "CaO":   {"Tm": 2845, "M": 56.08,  "V": 18.50, "alpha": 1.53, "type": "B",
              "mu0": [(1573, 0.03303), (1673, 0.02382), (1773, 0.01783)]},
    "MgO":   {"Tm": 3125, "M": 40.30,  "V": 16.50, "alpha": 1.51, "type": "B",
              "mu0": [(1573, 0.05651), (1673, 0.05966), (1773, 0.02897)]},
    "FeO":   {"Tm": 1650, "M": 71.85,  "V": 20.50, "alpha": 1.40, "type": "B",
              "mu0": [(1573, 0.00209), (1673, 0.00209), (1773, 0.00209)]},
    "Fe2O3": {"Tm": 1838, "M": 159.69, "V": 30.40, "alpha": 0.08, "type": "AM",
              "mu0": [(1573, 0.00086), (1673, 0.00080), (1773, 0.00073)]},
    "MnO":   {"Tm": 2115, "M": 70.94,  "V": 20.00, "alpha": 1.03, "type": "B",
              "mu0": [(1573, 0.00225), (1673, 0.00225), (1773, 0.00225)]},
    "K2O":   {"Tm": 1013, "M": 94.20,  "V": 36.00, "alpha": 1.37, "type": "B",
              "mu0": [(1573, 0.00050), (1673, 0.00050), (1773, 0.00050)]},
    "Na2O":  {"Tm": 1132, "M": 61.98,  "V": 20.60, "alpha": 1.94, "type": "B",
              "mu0": [(1573, 0.00060), (1673, 0.00060), (1773, 0.00060)]},
    "P2O5":  {"Tm": 843,  "M": 141.94, "V": 53.00, "alpha": 0.70, "type": "A",
              "mu0": [(1573, 0.00040), (1673, 0.00040), (1773, 0.00040)]},
    "TiO2":  {"Tm": 2130, "M": 79.87,  "V": 19.00, "alpha": 0.36, "type": "AM",
              "mu0": [(1573, 0.00300), (1673, 0.00250), (1773, 0.00200)]},
    "Cr2O3": {"Tm": 2708, "M": 151.99, "V": 44.30, "alpha": 0.71, "type": "AM",
              "mu0": [(1573, 0.01060), (1673, 0.00850), (1773, 0.00680)]},
    "CaF2":  {"Tm": 1691, "M": 78.07,  "V": 23.10, "alpha": 1.20, "type": "B",
              "mu0": [(1573, 0.00150), (1673, 0.00150), (1773, 0.00150)]},
}

# 光学碱度 Λ 各氧化物分量
OPTICAL_BASICITY = {
    "SiO2": 0.48, "CaO": 1.00, "MgO": 0.78, "Al2O3": 0.60,
    "FeO": 0.94, "Fe2O3": 0.72, "MnO": 0.95,
    "K2O": 1.40, "Na2O": 1.15, "TiO2": 0.61, "P2O5": 0.40,
    "Cr2O3": 0.63, "CaF2": 0.72,
}

# Iida 碱度转换系数
AB = (1.056, 0.0058, -1.115)

# 标定系数（针对高碱度钢铁渣，Bi*≥1.0 拟合）
A_SCALE  = 1.60
E_SCALE  = 0.65
MU0_SCALE = 1.40


# =====================================================================
# 内部工具函数
# =====================================================================
def _mass_to_mole(comp):
    """
    质量分数 → 摩尔分数
    返回 (x_mole_dict, w_mass_fraction_dict)
    """
    total = sum(comp.values())
    if total <= 0:
        raise ValueError("成分总和必须为正数")
    w = {k: v / total for k, v in comp.items()}
    # 过滤未知组分，避免 KeyError
    moles = {c: wf / COMPONENT_PARAMS[c]["M"]
             for c, wf in w.items() if c in COMPONENT_PARAMS}
    if not moles:
        raise ValueError("没有可识别的化学组分")
    tot = sum(moles.values())
    return {c: n / tot for c, n in moles.items()}, w


def _mu0i(T, c):
    """按温度线性插值得到组分 c 的理想黏度 μ₀i(T)"""
    p = COMPONENT_PARAMS[c]
    ts = [t[0] for t in p["mu0"]]
    vs = [t[1] for t in p["mu0"]]
    if T <= ts[0]:
        return vs[0]
    if T >= ts[-1]:
        return vs[-1]
    for i in range(len(ts) - 1):
        if ts[i] <= T <= ts[i + 1]:
            f = (T - ts[i]) / (ts[i + 1] - ts[i])
            return vs[i] + f * (vs[i + 1] - vs[i])
    return vs[-1]


# =====================================================================
# Iida 机理模型
# =====================================================================
def calc_iida_viscosity(comp, T_K):
    """
    Iida 模型黏度预测

    参数:
        comp: dict, 成分字典（wt%），如 {'SiO2':15, 'CaO':50, ...}
        T_K:  float, 绝对温度 (K)

    返回:
        dict: 包含 viscosity_Pa_s / A / E / mu0 / Bi_star / warnings
    """
    warnings_list = []

    total = sum(comp.values())
    if not (95 <= total <= 105):
        # 不再污染控制台，只走返回值
        warnings_list.append(
            f"成分总和 {total:.1f}% 超出 95-105%（建议归一化后输入）")

    W_FeO = comp.get("FeO", 0.0)
    if W_FeO > 20:
        warnings_list.append(
            "该渣系 FeO 含量超出 Iida 模型原始标定范围，结果仅供参考")

    # 摩尔分数 & 质量分数
    x, w = _mass_to_mole(comp)

    # 理想黏度 μ₀
    mu0 = MU0_SCALE * sum(_mu0i(T_K, c) * xi for c, xi in x.items())

    # 温度函数 A(T)、E(T)
    A = A_SCALE * (1.745 - 1.962e-3 * T_K + 7e-7 * T_K ** 2)
    E = E_SCALE * (11.11 - 3.65e-3 * T_K)

    # 碱度比 Bi_raw = nB / dB（B 组元 / A 组元）
    nb = sum(COMPONENT_PARAMS[c]["alpha"] * wf for c, wf in w.items()
             if c in COMPONENT_PARAMS and COMPONENT_PARAMS[c]["type"] == "B")
    db = sum(COMPONENT_PARAMS[c]["alpha"] * wf for c, wf in w.items()
             if c in COMPONENT_PARAMS and COMPONENT_PARAMS[c]["type"] == "A")
    Bi_raw = nb / db if db > 0 else 1.0

    # 修正 Al2O3 的等效系数 α*
    a_c, b_c, c_c = AB
    astar = max(a_c * Bi_raw + b_c * w.get("Al2O3", 0) + c_c, 0.05)

    # 修正后碱度 Bi*
    ns = 0.0
    ds = 0.0
    for c, wf in w.items():
        if c not in COMPONENT_PARAMS:
            continue
        p = COMPONENT_PARAMS[c]
        if c == "Al2O3":
            ds += astar * wf
        elif c == "Fe2O3":
            ns += p["alpha"] * wf
        elif p["type"] == "B":
            ns += p["alpha"] * wf
        elif p["type"] == "A":
            ds += p["alpha"] * wf

    Bi = ns / ds if ds > 0 else float('inf')

    if Bi <= 0 or math.isinf(Bi):
        warnings_list.append("Bi* 异常（ds=0 或 Bi≤0），无法计算黏度")
        return {
            "viscosity_Pa_s": float('nan'),
            "viscosity_Poise": float('nan'),
            "A": A, "E": E, "mu0": mu0,
            "Bi_star": Bi,
            "alpha_star_Al2O3": astar,
            "warnings": warnings_list,
        }

    try:
        visc = A * mu0 * math.exp(E / Bi)
        if visc > 1e6:
            visc = 1e6
            warnings_list.append("黏度过大已截断")
    except (OverflowError, ValueError):
        visc = 1e6
        warnings_list.append("黏度计算溢出，已截断")

    return {
        "viscosity_Pa_s": visc,
        "viscosity_Poise": visc * 10.0,
        "A": A,
        "E": E,
        "mu0": mu0,
        "Bi_star": Bi,
        "alpha_star_Al2O3": astar,
        "warnings": warnings_list,
    }


# =====================================================================
# 光学碱度
# =====================================================================
def calc_optical_basicity(comp):
    """
    光学碱度 Λ = Σ xᵢ·Λᵢ（xᵢ 为摩尔分数）
    对未列入 OPTICAL_BASICITY 的组分取中性 0.5
    """
    total = sum(comp.values())
    if total <= 0:
        raise ValueError("成分总和必须为正数")
    w = {k: v / total for k, v in comp.items()}
    moles = {c: wf / COMPONENT_PARAMS[c]["M"]
             for c, wf in w.items() if c in COMPONENT_PARAMS}
    if not moles:
        raise ValueError("没有可识别的化学组分")
    tot = sum(moles.values())
    x = {c: n / tot for c, n in moles.items()}
    return sum(xi * OPTICAL_BASICITY.get(c, 0.5) for c, xi in x.items())


# =====================================================================
# Gradient Boosting 数据驱动模型（嵌入）
# 训练数据: 244 条实测渣系黏度（钢铁渣 122 + 有色渣 122）
# 5 折 CV MAPE: 14.1%
# =====================================================================

_GB_MODEL = None
_GB_FEATURES = None


def _gb_model_path():
    """返回 GB 模型 pkl 的绝对路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'gb_viscosity_model.pkl')


def gb_model_available():
    """供 UI 判断 GB 模型 pkl 是否存在"""
    return os.path.exists(_gb_model_path())


def _load_gb_model():
    """惰性加载 GB 模型（首次调用时读 pkl）"""
    global _GB_MODEL, _GB_FEATURES
    if _GB_MODEL is None:
        pkl = _gb_model_path()
        if not os.path.exists(pkl):
            raise FileNotFoundError(
                f"未找到 GB 模型文件：{pkl}\n"
                f"请先训练模型，或将 gb_viscosity_model.pkl 放到程序目录下。")
        with open(pkl, 'rb') as f:
            pkg = pickle.load(f)
        _GB_MODEL = pkg['model']
        _GB_FEATURES = pkg['features']
    return _GB_MODEL, _GB_FEATURES


def calc_viscosity_ml(comp, temperature_K):
    """
    Gradient Boosting 数据驱动黏度预测

    参数:
        comp: dict, 成分字典（wt%），键为组分名（如 "SiO2"）
        temperature_K: float, 温度 (K)

    返回:
        dict: {
            "viscosity_Pa_s": 黏度 (Pa·s),
            "viscosity_Poise": 黏度 (Poise),
            "model": "GradientBoosting",
            "warnings": 警告列表
        }
    """
    model, features = _load_gb_model()
    warnings_list = []

    T_C = temperature_K - 273.15
    sio2  = comp.get('SiO2', 0)
    cao   = comp.get('CaO', 0)
    mgo   = comp.get('MgO', 0)
    al2o3 = comp.get('Al2O3', 0)
    feo   = comp.get('FeO', 0)
    fe2o3 = comp.get('Fe2O3', 0)
    mno   = comp.get('MnO', 0)
    c_s   = cao / sio2 if sio2 > 0 else 0

    feat = {
        'SiO2':  sio2,
        'CaO':   cao,
        'MgO':   mgo,
        'Al2O3': al2o3,
        'FeO':   feo,
        'Fe2O3': fe2o3,
        'MnO':   mno,
        'T_C':   T_C,
        'C/S':   c_s,
    }

    # 组装特征向量（严格按训练时的列顺序）
    x = np.array([[feat[f] for f in features]])
    log_eta = model.predict(x)[0]
    eta = float(10 ** log_eta)

    # 适用范围标注
    if feo > 20:
        warnings_list.append(
            "该渣系 FeO 含量较高，模型基于含 FeO 数据训练，结果仅供参考")
    if T_C < 1400 or T_C > 1700:
        warnings_list.append(
            f"温度 {T_C:.0f}°C 超出训练范围 (1400-1700°C)")
    if sio2 > 0 and (c_s < 0.5 or c_s > 5.0):
        warnings_list.append(
            f"碱度 C/S={c_s:.2f} 超出训练范围 (0.5-5.0)")

    return {
        "viscosity_Pa_s": eta,
        "viscosity_Poise": eta * 10.0,
        "model": "GradientBoosting",
        "warnings": warnings_list,
    }


# =====================================================================
# 简单的自测入口（直接运行本文件时执行）
# =====================================================================
if __name__ == "__main__":
    demo_comp = {
        'SiO2': 15.0, 'CaO': 50.0, 'Al2O3': 30.0, 'MgO': 5.0,
    }
    print("成分:", demo_comp)
    for tc in (1450, 1550, 1650):
        T = tc + 273.15
        ri = calc_iida_viscosity(demo_comp, T)
        print(f"T={tc}°C  Iida η = {ri['viscosity_Pa_s']:.4f} Pa·s  "
              f"(A={ri['A']:.3f}, E={ri['E']:.3f}, "
              f"μ₀={ri['mu0']:.5f}, Bi*={ri['Bi_star']:.3f})")
    print("光学碱度 Λ =", round(calc_optical_basicity(demo_comp), 4))

    if gb_model_available():
        for tc in (1450, 1550, 1650):
            T = tc + 273.15
            rg = calc_viscosity_ml(demo_comp, T)
            print(f"T={tc}°C  GB   η = {rg['viscosity_Pa_s']:.4f} Pa·s")
    else:
        print("（未找到 gb_viscosity_model.pkl，跳过 GB 模型自测）")