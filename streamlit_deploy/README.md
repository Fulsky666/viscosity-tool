# 跨渣系黏度交互式查询与对比工具

Cross-Slag Viscosity Interactive Tool

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 文件说明

- `app.py` — Streamlit 主程序
- `viscosity_calc.py` — Iida 机理模型 + GB 数据驱动模型
- `gb_viscosity_model.pkl` — 预训练 GB 模型
- `requirements.txt` — 依赖列表

## 模型

- **Iida**：Arrhenius 型半经验模型，适用于 CaO-SiO₂-Al₂O₃-MgO 钢铁精炼渣
- **GB**：Gradient Boosting 回归树，244 条实测数据训练，跨渣系预测
