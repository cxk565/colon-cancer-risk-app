import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.subplots as plt_subplots
import matplotlib.pyplot as plt
import os

# 强制后台绘图，防止网页端多线程冲突
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 0. 网页全局配置 & CSS 美化
# ==========================================
st.set_page_config(
    page_title="结肠癌术后低蛋白血症预警系统",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# 💉 注入自定义高级 CSS
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Times New Roman', 'Microsoft YaHei', sans-serif;
    }
    
    /* 核心主按钮样式重写 */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* 侧边栏背景 */
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E7E9;
    }
    
    /* 结果数值颜色强化 */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #C0392B; 
        font-weight: 900;
    }
    
    /* 输入框视觉强化 */
    input[type="number"] {
        font-weight: bold;
        color: #154360;
        background-color: #F4F6F7;
    }
    
    /* 隐藏不必要的原生组件 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 顶部抬头设计
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
with col_title:
    st.title("结肠癌患者术后低蛋白血症风险智能预警平台")
    st.markdown("**(Intelligent Warning Platform for Postoperative Hypoalbuminemia Risk in Colon Cancer)**")

st.markdown("""
<div style='background-color: #EBF5FB; padding: 15px; border-radius: 10px; border-left: 5px solid #2980B9; margin-bottom: 25px;'>
    <span style='color: #154360; font-size: 15px;'>
    <b>📊 系统简介：</b>本平台搭载 XGBoost 机器学习算法。通过整合患者术前及术中关键临床指标，<b>实时预警术后低蛋白血症</b>的发生风险。系统集成了 <b>SHAP 个体化解析技术</b>，旨在为临床医生提供直观、可解释的辅助决策支持。
    </span>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 2. 智能加载引擎 (直接读取同级目录)
# ==========================================
@st.cache_resource 
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "xgb_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error("🚨 模型文件加载失败，请检查文件是否已上传至 GitHub 根目录。")
    st.stop()


# ==========================================
# 3. 双向状态同步 (Two-way Binding)
# ==========================================
default_values = {
    'ChE': 6664.0, 'Age': 816.0, 'PA': 197.5, 
    'Crea': 69.7, 'FDP': 1.5, 'Lymph_pct': 24.2, 
    'CEA': 3.57, 'GLO': 28.6, 'Lymph_count': 1.59
}

for key, val in default_values.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val

def sync_inputs(src_key, dest_key):
    st.session_state[dest_key] = st.session_state[src_key]


# ==========================================
# 4. 侧边栏：快速滑动控制
# ==========================================
st.sidebar.markdown("### 🖥️ 系统运行状态")
st.sidebar.success("🟢 核心引擎: XGBoost 已就绪")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎛️ 临床指标快速调节")

with st.sidebar.expander("👤 人口学与肝肾功能", expanded=True):
    st.slider("年龄 (Age, 月龄)", 200.0, 1300.0, step=1.0, key="Age_slider", on_change=sync_inputs, args=("Age_slider", "Age_num"))
    st.slider("肌酐 (Crea) umol/L", 10.0, 1200.0, step=0.1, key="Crea_slider", on_change=sync_inputs, args=("Crea_slider", "Crea_num"))
    st.slider("前白蛋白 (PA) mg/L", 10.0, 800.0, step=1.0, key="PA_slider", on_change=sync_inputs, args=("PA_slider", "PA_num"))
    st.slider("球蛋白 (GLO) g/L", 10.0, 120.0, step=0.1, key="GLO_slider", on_change=sync_inputs, args=("GLO_slider", "GLO_num"))

with st.sidebar.expander("🩸 血液学指标", expanded=True):
    st.slider("淋巴细胞百分比 (Lymph%)", 0.0, 100.0, step=0.1, key="Lymph_pct_slider", on_change=sync_inputs, args=("Lymph_pct_slider", "Lymph_pct_num"))
    st.slider("淋巴细胞绝对值 (Lymph count)", 0.0, 50.0, step=0.01, key="Lymph_count_slider", on_change=sync_inputs, args=("Lymph_count_slider", "Lymph_count_num"))
    st.slider("纤维蛋白降解产物 (FDP) mg/L", 0.0, 300.0, step=0.01, key="FDP_slider", on_change=sync_inputs, args=("FDP_slider", "FDP_num"))
    
with st.sidebar.expander("🔬 特种酶与标志物", expanded=True):
    st.slider("胆碱酯酶 (ChE) U/L", 100.0, 25000.0, step=10.0, key="ChE_slider", on_change=sync_inputs, args=("ChE_slider", "ChE_num"))
    st.slider("癌胚抗原 (CEA) ng/mL", 0.0, 5000.0, step=0.1, key="CEA_slider", on_change=sync_inputs, args=("CEA_slider", "CEA_num"))


# ==========================================
# 5. 主界面：精确录入矩阵
# ==========================================
st.markdown("### 👨‍⚕️ 患者临床指标录入面板")
st.markdown("*(请根据化验单录入数值，系统将实时同步至左侧滑块)*")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.number_input("Age (月龄)", min_value=200.0, max_value=1300.0, step=1.0, format="%.0f", key="Age_num", on_change=sync_inputs, args=("Age_num", "Age_slider"))
    st.number_input("Crea (肌酐)", min_value=10.0, max_value=1200.0, step=0.1, format="%.1f", key="Crea_num", on_change=sync_inputs, args=("Crea_num", "Crea_slider"))
with col2:
    st.number_input("PA (前白蛋白)", min_value=10.0, max_value=800.0, step=1.0, format="%.1f", key="PA_num", on_change=sync_inputs, args=("PA_num", "PA_slider"))
    st.number_input("GLO (球蛋白)", min_value=10.0, max_value=120.0, step=0.1, format="%.1f", key="GLO_num", on_change=sync_inputs, args=("GLO_num", "GLO_slider"))
with col3:
    st.number_input("Lymph% (淋巴%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f", key="Lymph_pct_num", on_change=sync_inputs, args=("Lymph_pct_num", "Lymph_pct_slider"))
    st.number_input("Lymph count (淋巴绝对值)", min_value=0.0, max_value=50.0, step=0.01, format="%.2f", key="Lymph_count_num", on_change=sync_inputs, args=("Lymph_count_num", "Lymph_count_slider"))
with col4:
    st.number_input("ChE (胆碱酯酶)", min_value=100.0, max_value=25000.0, step=10.0, format="%.0f", key="ChE_num", on_change=sync_inputs, args=("ChE_num", "ChE_slider"))
    st.number_input("CEA (癌胚抗原)", min_value=0.0, max_value=5000.0, step=0.1, format="%.2f", key="CEA_num", on_change=sync_inputs, args=("CEA_num", "CEA_slider"))

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.number_input("FDP (FDP含量)", min_value=0.0, max_value=300.0, step=0.01, format="%.2f", key="FDP_num", on_change=sync_inputs, args=("FDP_num", "FDP_slider"))


# 构造特征向量
input_df = pd.DataFrame({
    'ChE': [st.session_state["ChE_num"]], 
    'Age': [st.session_state["Age_num"]], 
    'PA': [st.session_state["PA_num"]], 
    'Crea': [st.session_state["Crea_num"]],
    'FDP': [st.session_state["FDP_num"]], 
    'Lymph%': [st.session_state["Lymph_pct_num"]], 
    'CEA': [st.session_state["CEA_num"]], 
    'GLO': [st.session_state["GLO_num"]],
    'Lymphocyte count': [st.session_state["Lymph_count_num"]] 
})

# ==========================================
# 6. 核心预测与解释引擎
# ==========================================
if st.button("🚀 开启 AI 智能预警推演", type="primary"):
    with st.spinner('🧬 正在运行多维特征解析...'):
        
        # 风险概率预测
        risk_prob = model.predict_proba(input_df)[0][1] 
        
        st.markdown("---")
        st.markdown("### 🎯 术后风险推断报告")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="术后低蛋白血症概率", value=f"{risk_prob * 100:.2f} %")
            
        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            if risk_prob > 0.5: 
                st.error("🚨 **【红色预警】该患者处于高风险区间！** 模型判定该患者极易发生**术后低蛋白血症**。建议加强围手术期营养管理，并考虑术前补充白蛋白及加强术后监测。")
                st.toast('检测到高危预警！', icon='⚠️') 
            else:
                st.success("✅ **【安全评估】该患者处于低风险区间。** 暂未检测到显著的低蛋白血症发生倾向，建议维持常规术后营养方案。")
                st.balloons() 

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 风险动因溯源 (SHAP 个体化靶点解析)")
        st.info("💡 **可视化解读：** 红色条柱代表将风险推高的**危险因子**，蓝色条柱代表**保护因子**。条柱越长，影响权重越大。")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(input_df)
            
            # 处理 SHAP 数据格式
            if isinstance(shap_values_raw, list):
                shap_val_single = shap_values_raw[1][0]
            elif len(np.array(shap_values_raw).shape) == 3:
                shap_val_single = shap_values_raw[0, :, 1]
            else:
                shap_val_single = shap_values_raw[0]
                
            ev = explainer.expected_value
            base_val = ev[1] if isinstance(ev, (list, np.ndarray)) and len(np.array(ev).flatten()) > 1 else (ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
            
            # 渲染瀑布图
            fig, ax = plt.subplots(figsize=(10, 6))
            exp = shap.Explanation(values=shap_val_single, base_values=base_val, 
                                   data=input_df.iloc[0], feature_names=input_df.columns.tolist())
            shap.waterfall_plot(exp, max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig) 
            
        except Exception as e:
            st.error(f"解析瀑布图时遭遇异常: {e}")
