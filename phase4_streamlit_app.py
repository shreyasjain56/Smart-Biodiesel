import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Smart Biodiesel Optimizer", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    with open('biodiesel_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

try:
    MODEL_FEATURES = list(model.feature_names_in_)
except AttributeError:
    MODEL_FEATURES = ['temperature', 'molar_ratio', 'catalyst_conc', 'reaction_time']

def make_input_df(temp, ratio, cat, time):
    return pd.DataFrame([[temp, ratio, cat, time]], columns=MODEL_FEATURES)

def make_grid_df(T_flat, R_flat, cat, time):
    n = len(T_flat)
    return pd.DataFrame(
        np.column_stack([T_flat, R_flat, np.full(n, cat), np.full(n, time)]),
        columns=MODEL_FEATURES
    )


st.markdown("""
<h1 style='text-align:center;color:#1D9E75;margin-bottom:4px'>
🌿 Smart Biodiesel Process Optimizer</h1>
<p style='text-align:center;color:#888;font-size:16px;margin-top:0'>
AI-powered FAME yield prediction · Dept. of Chemical Engineering · MNIT Jaipur</p>
<hr style='border-color:#E1F5EE'>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.subheader("Set process parameters")
    temperature   = st.slider("Reaction temperature (°C)",    40.0, 65.0,  57.0, 0.5)
    molar_ratio   = st.slider("Methanol:oil molar ratio",      3.0,  9.0,   6.4, 0.1)
    catalyst_conc = st.slider("Catalyst concentration (wt%)", 0.5,  1.5,   0.95, 0.05)
    reaction_time = st.slider("Reaction time (minutes)",       30,   120,    90,   5)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Predict FAME yield", type="primary", use_container_width=True)

with col_right:
    predicted_yield = model.predict(make_input_df(temperature, molar_ratio, catalyst_conc, reaction_time))[0]

    if predicted_yield >= 90:   yield_color, quality = "#1D9E75", "Excellent"
    elif predicted_yield >= 80: yield_color, quality = "#BA7517", "Good"
    elif predicted_yield >= 70: yield_color, quality = "#D85A30", "Moderate"
    else:                       yield_color, quality = "#A32D2D", "Poor"

    st.markdown(f"""
    <div style='background:#F8FFFE;border:2px solid {yield_color};border-radius:12px;
                padding:24px;text-align:center;margin-bottom:16px'>
        <p style='color:#888;font-size:14px;margin:0 0 4px 0'>Predicted FAME Yield</p>
        <h1 style='color:{yield_color};font-size:56px;margin:0;font-weight:700'>{predicted_yield:.1f}%</h1>
        <p style='color:{yield_color};font-size:18px;margin:4px 0 0 0;font-weight:500'>{quality}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Your input conditions:**")
    st.dataframe(pd.DataFrame({
        'Parameter':     ['Temperature','Molar ratio','Catalyst conc.','Reaction time'],
        'Your value':    [f"{temperature} °C", f"{molar_ratio:.1f} : 1",
                          f"{catalyst_conc:.2f} wt%", f"{reaction_time} min"],
        'Optimal value': ["57 °C", "6.4 : 1", "0.95 wt%", "90 min"]
    }), hide_index=True, use_container_width=True)

    st.markdown("**Yield sensitivity map:**")
    try:
        temp_range  = np.linspace(40.0, 65.0, 60)
        ratio_range = np.linspace(3.0,  9.0,  60)
        TG, RG = np.meshgrid(temp_range, ratio_range)
        
        Z = model.predict(
            make_grid_df(TG.ravel(), RG.ravel(), catalyst_conc, reaction_time)
        ).reshape(TG.shape)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        fig.subplots_adjust(left=0.12, right=0.82, top=0.88, bottom=0.14)

        hm = ax.contourf(TG, RG, Z, levels=25, cmap='RdYlGn', vmin=55, vmax=98)

        cax = fig.add_axes([0.84, 0.14, 0.03, 0.74]) 
        cbar = fig.colorbar(hm, cax=cax)
        cbar.set_label('FAME yield (%)', fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)

        ax.scatter(temperature, molar_ratio, color='white', s=220, zorder=5,
                   marker='*', edgecolors='black', linewidths=1.2)
        ax.annotate(f'{predicted_yield:.1f}%',
                    xy=(temperature, molar_ratio),
                    xytext=(temperature + 1.0, molar_ratio + 0.3),
                    fontsize=9, color='white', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.8))

        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('Molar ratio', fontsize=10)
        ax.set_title('Yield surface map  |  star = your conditions', fontsize=10, pad=8)
        ax.tick_params(labelsize=9)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"Heatmap error: {e}")

st.markdown(f"""<hr style='border-color:#E1F5EE;margin-top:32px'>
<p style='text-align:center;color:#aaa;font-size:12px'>
Model: Random Forest (200 trees) · R²=0.89 · RMSE=3.22% ·
Features: {', '.join(MODEL_FEATURES)} · Dept. of Chemical Technology, MNIT Jaipur
</p>""", unsafe_allow_html=True)