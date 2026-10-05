# -*- coding: utf-8 -*-
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from viscosity_calc import (calc_iida_viscosity, calc_viscosity_ml,
                            calc_optical_basicity, gb_model_available)

st.set_page_config(page_title="跨渣系黏度查询工具", page_icon="⚗",
                   layout="wide", initial_sidebar_state="expanded")

# ===== 语言切换 =====
if "lang" not in st.session_state:
    st.session_state.lang = "中文"
_hc1, _hc2 = st.columns([5, 1])
with _hc2:
    st.radio("语言 / Language", ["中文", "English"], horizontal=True, key="lang")
LANG = "en" if st.session_state.lang == "English" else "zh"
with _hc1:
    st.markdown("## ⚗ 跨渣系黏度交互式查询与对比工具" if LANG=="zh"
                else "## ⚗ Cross-Slag Viscosity Interactive Tool")
st.markdown("---")

# ===== 翻译字典 =====
T = {
    "钢铁渣系": ("钢铁渣系", "Steel slag"), "铜渣": ("铜渣", "Copper slag"),
    "镍渣": ("镍渣", "Nickel slag"), "自定义": ("自定义", "Custom"),
    "参数设置": ("参数设置", "Parameters"), "渣系类型": ("渣系类型", "Slag type"),
    "化学成分": ("化学成分 (wt%)", "Composition (wt%)"),
    "更多组分": ("更多组分", "More"), "成分合计": ("成分合计", "Total"),
    "温度": ("温度 (°C)", "Temperature (°C)"),
    "显示曲线": ("显示温度-黏度曲线", "Show curve"),
    "对比模型": ("对比 Iida 与 GB", "Compare Iida & GB"),
    "上传实测": ("上传实测CSV", "Upload measured CSV"),
    "批量预测": ("批量预测：上传成分表", "Batch prediction"),
    "计算结果": ("计算结果", "Results"),
    "Iida预测": ("Iida模型预测", "Iida prediction"),
    "GB预测": ("GB模型预测", "GB prediction"),
    "光学碱度": ("光学碱度 Λ", "Optical basicity"),
    "碱度CS": ("碱度 C/S", "C/S ratio"),
    "黏度曲线": ("📊 黏度曲线", "📊 Curve"),
    "中间参数": ("📋 中间参数", "📋 Parameters"),
    "模型说明": ("📚 模型说明", "📚 Models"),
    "使用指南": ("📖 使用指南", "📖 Guide"),
    "温度轴": ("温度 (°C)", "Temperature (°C)"),
    "黏度轴": ("黏度 (Pa·s)", "Viscosity (Pa·s)"),
    "黑色线": ("黑色实线：Iida机理模型", "Black: Iida model"),
    "红色线": ("红色实线：GB数据驱动模型", "Red: GB model"),
    "下载曲线": ("📥 下载曲线数据", "📥 Download curve"),
    "下载批量": ("📥 下载结果", "📥 Download results"),
}
def t(key):
    v = T.get(key, (key, key))
    return v[1] if LANG == "en" else v[0]

# ===== 预设渣系 =====
PRESETS = {
    "钢铁渣系": {'SiO2':15,'CaO':50,'Al2O3':30,'MgO':5,'FeO':0,'MnO':0,'Fe2O3':0,'TiO2':0,'K2O':0,'Na2O':0},
    "铜渣":     {'SiO2':30,'CaO':10,'Al2O3':5,'MgO':3,'FeO':45,'MnO':0,'Fe2O3':5,'TiO2':0,'K2O':0,'Na2O':0},
    "镍渣":     {'SiO2':25,'CaO':15,'Al2O3':5,'MgO':5,'FeO':40,'MnO':0,'Fe2O3':3,'TiO2':0,'K2O':0,'Na2O':0},
    "自定义":   {'SiO2':25,'CaO':35,'Al2O3':15,'MgO':5,'FeO':5,'MnO':0,'Fe2O3':0,'TiO2':0,'K2O':0,'Na2O':0},
}
KEYS = list(PRESETS["钢铁渣系"].keys())

# ===== 侧边栏 =====
with st.sidebar:
    st.header(t("参数设置"))
    if "slag" not in st.session_state:
        st.session_state.slag = "钢铁渣系"
        for k in KEYS:
            st.session_state[f"in_{k}"] = PRESETS["钢铁渣系"][k]

    def _preset():
        for k in KEYS:
            st.session_state[f"in_{k}"] = PRESETS[st.session_state.slag][k]

    slag_display = [t(k) for k in PRESETS.keys()]
    def _on_slang():
        idx = slag_display.index(st.session_state.slag_sel)
        st.session_state.slag = list(PRESETS.keys())[idx]
        _preset()
    st.selectbox(t("渣系类型"), slag_display,
                 index=slag_display.index(t(st.session_state.slag)),
                 key="slag_sel", on_change=_on_slang)

    st.subheader(t("化学成分"))
    c1, c2 = st.columns(2)
    with c1:
        sio2  = st.number_input("SiO₂",  0.0,100.0,step=0.01,format="%.2f",key="in_SiO2")
        cao   = st.number_input("CaO",   0.0,100.0,step=0.01,format="%.2f",key="in_CaO")
        al2o3 = st.number_input("Al₂O₃", 0.0,100.0,step=0.01,format="%.2f",key="in_Al2O3")
    with c2:
        mgo = st.number_input("MgO", 0.0,100.0,step=0.01,format="%.2f",key="in_MgO")
        feo = st.number_input("FeO", 0.0,100.0,step=0.01,format="%.2f",key="in_FeO")
        mno = st.number_input("MnO", 0.0,100.0,step=0.01,format="%.2f",key="in_MnO")
    with st.expander(t("更多组分")):
        ce1, ce2 = st.columns(2)
        with ce1:
            fe2o3 = st.number_input("Fe₂O₃",0.0,100.0,step=0.01,format="%.2f",key="in_Fe2O3")
            tio2  = st.number_input("TiO₂", 0.0,100.0,step=0.01,format="%.2f",key="in_TiO2")
        with ce2:
            k2o  = st.number_input("K₂O",0.0,100.0,step=0.01,format="%.2f",key="in_K2O")
            na2o = st.number_input("Na₂O",0.0,100.0,step=0.01,format="%.2f",key="in_Na2O")

    comp_raw = {'SiO2':sio2,'CaO':cao,'Al2O3':al2o3,'MgO':mgo,
                'FeO':feo,'MnO':mno,'Fe2O3':fe2o3,'TiO2':tio2,
                'K2O':k2o,'Na2O':na2o}
    total_raw = sum(comp_raw.values())
    st.caption(f"{t('成分合计')}: **{total_raw:.1f}%**")
    temp = st.number_input(t("温度"), 1000, 1800, 1550, 10)
    show_curve = st.checkbox(t("显示曲线"), True)
    gb_ok = gb_model_available()
    show_gb = st.checkbox(t("对比模型"), gb_ok, disabled=not gb_ok)
    exp_file = st.file_uploader(t("上传实测"), type=["csv"])
    st.markdown("---")
    batch_file = st.file_uploader(t("批量预测"), type=["xlsx","xls","csv"])

# ===== 输入校验 =====
if total_raw <= 0:
    st.error("成分合计必须大于0 / Total must be > 0"); st.stop()
if abs(total_raw - 100) > 1:
    st.info("成分合计不为100%，已自动归一化" if LANG=="zh"
            else "Composition not 100%, auto-normalized")
comp = {k: v*100/total_raw for k, v in comp_raw.items() if v > 0}
T_K = temp + 273.15

# ===== 实测数据 =====
exp_df = None
if exp_file is not None:
    try:
        exp_df = pd.read_csv(exp_file)
        exp_df.columns = [c.strip().lower() for c in exp_df.columns]
        exp_df = exp_df.rename(columns={'temp':'T_C','temperature':'T_C',
                'viscosity_pa_s':'eta_Pa_s','viscosity':'eta_Pa_s'})
        if 'T_C' not in exp_df or 'eta_Pa_s' not in exp_df:
            st.warning("CSV需含温度和黏度两列 / CSV needs T and viscosity columns")
            exp_df = None
    except Exception as e:
        st.warning(f"读取失败 / Read error: {e}"); exp_df = None

# ===== 单条计算 =====
try:
    ri = calc_iida_viscosity(comp, T_K)
    eta_i, Bi, A_v, E_v, mu0 = ri['viscosity_Pa_s'], ri['Bi_star'], ri['A'], ri['E'], ri['mu0']
except Exception as e:
    eta_i, Bi, A_v, E_v, mu0 = np.nan, 0, 0, 0, 0
    st.error(f"Iida计算错误 / Iida error: {e}")

eta_g = np.nan
if gb_ok:
    try:
        rg = calc_viscosity_ml(comp, T_K)
        eta_g = rg['viscosity_Pa_s']
        for w in rg.get('warnings', []):
            if LANG == 'en':
                if '温度' in w and '超出' in w:
                    w = "Temperature outside training range (1400-1700°C)"
                elif '碱度' in w and '超出' in w:
                    m = re.search(r'C/S=([\d.]+)', w)
                    w = f"C/S={m.group(1) if m else '?'} outside range (0.5-5.0)"
                elif 'FeO' in w and '超出' in w:
                    w = "High FeO outside Iida calibration, reference only"
                elif 'FeO' in w:
                    w = "High FeO: model trained on FeO data, reference only"
            st.warning(w)
    except Exception as e:
        st.error(f"GB计算错误 / GB error: {e}")

try:
    Lambda = calc_optical_basicity(comp)
except Exception:
    Lambda = np.nan

# ===== 结果卡片 =====
st.subheader(t("计算结果"))
cols = st.columns(4 if not show_gb else 5)
cols[0].metric(t("Iida预测"), f"{eta_i:.4f} Pa·s" if not np.isnan(eta_i) else "N/A",
               delta=f"Bi*={Bi:.2f}")
if show_gb:
    cols[1].metric(t("GB预测"), f"{eta_g:.4f} Pa·s" if not np.isnan(eta_g) else "N/A")
    j = 2
else:
    j = 1
cols[j].metric(t("光学碱度"), f"{Lambda:.3f}" if not np.isnan(Lambda) else "N/A")
cols[j+1].metric(t("碱度CS"), f"{cao/sio2:.2f}" if sio2 > 0 else "N/A")

# ===== 标签页 =====
tab1, tab2, tab3, tab4 = st.tabs([t("黏度曲线"), t("中间参数"), t("模型说明"), t("使用指南")])

@st.cache_data(show_spinner=False)
def _curve_cached(comp_tuple, ts_tuple, gb_flag):
    d = dict(comp_tuple)
    out_i, out_g = [], []
    for tt in ts_tuple:
        try: out_i.append(calc_iida_viscosity(d, tt+273.15)['viscosity_Pa_s'])
        except Exception: out_i.append(np.nan)
        if gb_flag:
            try: out_g.append(calc_viscosity_ml(d, tt+273.15)['viscosity_Pa_s'])
            except Exception: out_g.append(np.nan)
        else:
            out_g.append(np.nan)
    return out_i, out_g

with tab1:
    if show_curve:
        lo, hi = max(1100, temp-200), min(1750, temp+200)
        ts = np.arange(lo, hi+1, 5)
        ic, gc = _curve_cached(tuple(sorted(comp.items())), tuple(ts), show_gb)
        ia, ga = np.array(ic), np.array(gc)
        all_vals = ia if not show_gb else np.concatenate([ia, ga[~np.isnan(ga)]])
        y_min = max(0.01, np.nanmin(all_vals) * 0.8)
        y_max = np.nanmax(all_vals) * 1.2

        st.markdown(
            "<div style='margin-left:55px;margin-right:40px;margin-bottom:5px;"
            "font-size:14px;text-align:right;background:#333;"
            "padding:6px 10px;border-radius:3px;'>"
            f"<b style='color:white;'>{t('黑色线')}</b>"
            "&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<b style='color:white;'>{t('红色线')}</b></div>",
            unsafe_allow_html=True)

        ht = '温度 %{x}°C | 黏度 %{y:.4f} Pa·s<extra></extra>' if LANG=='zh' \
             else 'T %{x}°C | η %{y:.4f} Pa·s<extra></extra>'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=ia, mode='lines+markers', name='Iida',
            line=dict(color='black', width=2),
            marker=dict(size=6, color='white', line=dict(color='black', width=1.5)),
            hovertemplate=ht))
        if show_gb:
            fig.add_trace(go.Scatter(x=ts, y=ga, mode='lines+markers', name='GB',
                line=dict(color='#c0392b', width=2),
                marker=dict(size=6, color='white', line=dict(color='#c0392b', width=1.5)),
                hovertemplate=ht))
        fig.add_vline(x=temp, line_dash='dot', line_color='gray')
        if exp_df is not None:
            fig.add_trace(go.Scatter(x=exp_df['T_C'], y=exp_df['eta_Pa_s'],
                mode='markers', name='实测',
                marker=dict(symbol='triangle-up', size=10, color='green',
                            line=dict(color='black', width=1)),
                hovertemplate=ht))

        fig.update_layout(
            xaxis=dict(title=t("温度轴"), range=[lo, hi], dtick=50,
                       showgrid=True, gridcolor='#e0e0e0',
                       ticks='inside', mirror=True, linewidth=1.2, linecolor='black'),
            yaxis=dict(title=t("黏度轴"), type='log',
                       range=[np.log10(y_min), np.log10(y_max)],
                       showgrid=True, gridcolor='#e0e0e0',
                       ticks='inside', mirror=True, linewidth=1.2, linecolor='black'),
            showlegend=False, hovermode='x unified', plot_bgcolor='white',
            margin=dict(l=70, r=40, t=20, b=50), height=500, font=dict(size=13))
        st.plotly_chart(fig, use_container_width=True)

        tcol = '温度°C' if LANG=='zh' else 'T(°C)'
        icol = 'Iida预测(Pa·s)' if LANG=='zh' else 'Iida(Pa·s)'
        gcol = 'GB预测(Pa·s)' if LANG=='zh' else 'GB(Pa·s)'
        df = pd.DataFrame({tcol: ts,
            icol: [round(v,4) if not np.isnan(v) else None for v in ic],
            gcol: [round(v,4) if not np.isnan(v) else None for v in gc]})
        st.dataframe(df, hide_index=True, use_container_width=True, height=250)
        st.download_button(t("下载曲线"),
            df.to_csv(index=False).encode('utf-8-sig'),
            "viscosity_curve.csv", "text/csv")

with tab2:
    ca, cb = st.columns(2)
    with ca:
        if LANG == 'zh':
            st.write(f"**温度函数 A = {A_v:.4f}**")
            st.write(f"**温度函数 E = {E_v:.4f}**")
            st.write(f"**理想黏度 μ₀ = {mu0:.6f} Pa·s**")
            st.write(f"**修正碱度指数 Bi* = {Bi:.4f}**")
        else:
            st.write(f"**A = {A_v:.4f}**")
            st.write(f"**E = {E_v:.4f}**")
            st.write(f"**μ₀ = {mu0:.6f} Pa·s**")
            st.write(f"**Bi* = {Bi:.4f}**")
    with cb:
        st.dataframe(pd.DataFrame({
            '组分' if LANG=='zh' else 'Component': list(comp.keys()),
            '质量分数%' if LANG=='zh' else 'wt%': [round(v,2) for v in comp.values()]}),
            hide_index=True, use_container_width=True)

with tab3:
    st.subheader(t("模型说明"))
    c1, c2 = st.columns(2)
    with c1:
        if LANG == 'zh':
            st.markdown("""**Iida机理模型**

基于Arrhenius方程的半经验模型
μ = A(T)·μ₀·exp(E(T)/Bi*)

- 适用于CaO-SiO₂-Al₂O₃-MgO钢铁精炼渣
- 高碱度钢铁渣平均相对误差约35%
- 高FeO有色渣偏差较大""")
        else:
            st.markdown("""**Iida model**

Arrhenius-type semi-empirical model
μ = A(T)·μ₀·exp(E(T)/Bi*)

- For CaO-SiO₂-Al₂O₃-MgO steel refining slag
- MAPE ≈ 35% for high-basicity steel slag
- Larger deviation for high-FeO non-ferrous slag""")
    with c2:
        if LANG == 'zh':
            st.markdown("""**GB数据驱动模型**

梯度提升回归树（Gradient Boosting）

- 244条实测数据训练（钢铁渣122+有色渣122）
- 5折交叉验证平均相对误差14.1%
- 跨渣系预测稳定""")
        else:
            st.markdown("""**GB model**

Gradient Boosting Regressor

- Trained on 244 measured datasets
- 5-fold CV MAPE = 14.1%
- Stable across slag types""")
    st.divider()
    st.markdown("**使用建议**：钢铁精炼渣两个模型均可参考；高FeO铜渣/镍渣以GB为准；温度超出1400-1700°C时结果仅供参考"
                if LANG=="zh" else
                "**Note**: Use GB for high-FeO copper/nickel slag; results outside 1400-1700°C are for reference only")

with tab4:
    if LANG == 'zh':
        st.markdown("""
## 使用指南

### 一、快速上手
1. **选择渣系类型**：左侧下拉框选钢铁/铜/镍/自定义，成分自动填入
2. **调整成分**：在输入框直接修改各氧化物质量分数（%），精确到0.01
3. **设置温度**：输入目标温度（°C）
4. **查看结果**：页面顶部显示Iida和GB两个模型的预测黏度

### 二、黏度曲线图
- 黑色实线 = Iida机理模型，红色实线 = GB数据驱动模型
- 鼠标悬停在曲线上可实时查看任意温度下的黏度值
- 灰色虚线标记当前输入温度

### 三、批量预测
- 左侧上传成分表：Excel或CSV文件
- 需有温度列，其余列为各氧化物质量分数
- 自动识别SiO₂、CaO、Al₂O₃、MgO、FeO等组分
- 上传后自动逐行计算，结果可下载

### 四、模型选择建议
| 渣系类型 | 推荐模型 | 说明 |
|---|---|---|
| 钢铁精炼渣 | 两个均可参考 | Iida约35%误差，GB约14% |
| 铜渣/镍渣（高FeO>20%） | 以GB为准 | Iida在高FeO区偏差大 |
| 温度<1400或>1700°C | 仅供参考 | 超出模型训练范围 |

### 五、文件要求
- 成分总和不要求=100%，系统自动归一化
""")
    else:
        st.markdown("""
## User Guide

### 1. Quick Start
1. **Select slag type**: choose Steel/Copper/Nickel/Custom from dropdown
2. **Adjust composition**: enter oxide wt% (precision 0.01)
3. **Set temperature**: input target T (°C)
4. **View results**: Iida and GB predictions shown at top

### 2. Viscosity Curve
- Black line = Iida model, Red line = GB model
- Hover to see viscosity at any temperature
- Gray dashed line marks current temperature

### 3. Batch Prediction
- Upload Excel/CSV with temperature column and composition columns
- Auto-detects SiO₂, CaO, Al₂O₃, MgO, FeO, etc.
- Results table can be downloaded as CSV

### 4. Model Selection
| Slag type | Recommended | Notes |
|---|---|---|
| Steel refining slag | Both | Iida ~35% error, GB ~14% |
| Copper/nickel slag (FeO>20%) | GB only | Iida deviates at high FeO |
| T <1400 or >1700°C | Reference only | Outside training range |

### 5. Notes
- Composition sum need not equal 100%, auto-normalized
""")

# ===== 批量预测 =====
def _norm_col(name):
    n = str(name).strip().replace('%','').replace(' ','').lower()
    for a,b in [('₂','2'),('₃','3'),('氧化硅','sio2'),('二氧化硅','sio2'),
                ('氧化钙','cao'),('氧化镁','mgo'),('氧化铝','al2o3'),
                ('氧化亚铁','feo'),('三氧化二铁','fe2o3'),('氧化锰','mno'),
                ('二氧化钛','tio2'),('氧化钠','na2o'),('氧化钾','k2o')]:
        n = n.replace(a,b)
    return n

def _match_comp(name):
    n = _norm_col(name)
    for key, kw in [('SiO2','sio'),('CaO','cao'),('MgO','mgo'),('Al2O3','al2'),
                    ('Fe2O3','fe2'),('FeO','feo'),('MnO','mno'),('TiO2','tio'),
                    ('Na2O','na2'),('K2O','k2'),('Cr2O3','cr2'),('CaF2','caf')]:
        if kw in n: return key
    return None

if batch_file is not None:
    st.markdown("---")
    st.subheader("📋 批量预测结果" if LANG=='zh' else "📋 Batch Results")
    try:
        df_in = pd.read_excel(batch_file) if not batch_file.name.endswith('.csv') else pd.read_csv(batch_file)
        df_in.columns = [str(c).strip() for c in df_in.columns]

        SKIP = ['编号','数据集','渣系','类型','实测','预测','误差','备注','note','温度','viscosity']

        # 找关键列
        temp_col = next((c for c in df_in.columns
                         if '温度' in _norm_col(c) or 'temp' in _norm_col(c)), None)
        eta_col = next((c for c in df_in.columns
                        if any(k in _norm_col(c) for k in ['实测','viscosity','黏度','粘度'])
                        and '预测' not in _norm_col(c) and '误差' not in _norm_col(c)), None)
        id_col = next((c for c in df_in.columns
                       if '编号' in str(c) or str(c).lower() in ('id','no.','no','序号')), None)

        if temp_col is None:
            st.error("未找到温度列 / Temperature column not found"); st.stop()

        col_map = {}
        for c in df_in.columns:
            if c in (temp_col, eta_col, id_col): continue
            cn = _norm_col(c)
            if any(w in cn for w in SKIP) or cn in ('cs','c/s','碱度'): continue
            key = _match_comp(c)
            if key: col_map[c] = key

        st.caption(f"组分映射: {col_map}" if LANG=='zh' else f"Component mapping: {col_map}")

        labels = {'编号':'编号','id':'ID','温度°C':'温度°C','T(°C)':'T(°C)',
                  '实测黏度(Pa·s)':'实测黏度(Pa·s)','Measured(Pa·s)':'Measured(Pa·s)',
                  'Iida预测(Pa·s)':'Iida预测(Pa·s)','Iida(Pa·s)':'Iida(Pa·s)',
                  'GB预测(Pa·s)':'GB预测(Pa·s)','GB(Pa·s)':'GB(Pa·s)',
                  '错误':'错误','Error':'Error'}
        id_l = labels['编号'] if LANG=='zh' else labels['id']
        t_l = labels['温度°C'] if LANG=='zh' else labels['T(°C)']
        eta_l = labels['实测黏度(Pa·s)'] if LANG=='zh' else labels['Measured(Pa·s)']
        iida_l = labels['Iida预测(Pa·s)'] if LANG=='zh' else labels['Iida(Pa·s)']
        gb_l = labels['GB预测(Pa·s)'] if LANG=='zh' else labels['GB(Pa·s)']
        err_l = labels['错误'] if LANG=='zh' else labels['Error']

        results = []
        for _, row in df_in.iterrows():
            try:
                Tc = float(row[temp_col])
                cr = {}
                for orig, key in col_map.items():
                    try:
                        v = float(row[orig])
                        if v > 0: cr[key] = v
                    except (ValueError, TypeError): pass
                if not cr: continue
                ri = calc_iida_viscosity(cr, Tc+273.15)
                rr = {}
                if id_col: rr[id_l] = row[id_col]
                rr[t_l] = Tc
                if eta_col:
                    try: rr[eta_l] = round(float(row[eta_col]), 4)
                    except: pass
                rr[iida_l] = round(ri['viscosity_Pa_s'], 4)
                if gb_ok:
                    rg = calc_viscosity_ml(cr, Tc+273.15)
                    rr[gb_l] = round(rg['viscosity_Pa_s'], 4)
                results.append(rr)
            except Exception as e:
                rr = {}
                if id_col: rr[id_l] = row[id_col]
                rr[t_l] = row[temp_col]
                rr[err_l] = str(e)
                results.append(rr)

        if not results:
            st.warning("没有可计算的行 / No valid rows")
        else:
            df_out = pd.DataFrame(results)
            st.dataframe(df_out, hide_index=True, use_container_width=True)
            st.download_button(t("下载批量"),
                df_out.to_csv(index=False).encode('utf-8-sig'),
                "batch_prediction.csv", "text/csv")
    except Exception as e:
        st.error(f"读取文件失败 / Read error: {e}")
