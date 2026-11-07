"""
Planetary Defense Analytics — Streamlit app
Run locally from the project folder:  streamlit run app/app.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
AU_TO_LD = 389.17  # 1 astronomical unit = 389.17 lunar distances

st.set_page_config(page_title="Planetary Defense Analytics", page_icon="☄️", layout="wide")


# ---------- Load models ----------
@st.cache_resource
def load_models():
    clf = joblib.load(MODELS_DIR / "pha_classifier.joblib")
    reg = joblib.load(MODELS_DIR / "diameter_regressor.joblib")
    clu = joblib.load(MODELS_DIR / "cluster_model.joblib")
    return clf, reg, clu


@st.cache_resource
def get_explainer(_artifact, key):
    """SHAP explainer for a saved pipeline (cached per model)."""
    pipe = _artifact["pipeline"]
    model = pipe.named_steps["model"]
    if isinstance(model, (LogisticRegression, LinearRegression)):
        bg = _artifact["background"]
        if "scaler" in pipe.named_steps:
            bg = pd.DataFrame(pipe.named_steps["scaler"].transform(bg), columns=bg.columns)
        return shap.LinearExplainer(model, bg)
    return shap.TreeExplainer(model)


clf_art, reg_art, clu_art = load_models()


# ---------- Feature engineering (must match the notebooks) ----------
def engineer(raw: dict) -> pd.DataFrame:
    a, e = raw["a"], raw["e"]
    f = {
        "H": raw["H"],
        "e": e,
        "i": raw["i"],
        "log_a": np.log10(a),
        "log_n": np.log10(0.9856076686 / a ** 1.5),   # mean motion (deg/day) from Kepler's 3rd law
        "log_per_y": np.log10(a ** 1.5),               # period (years) from Kepler's 3rd law
        "log_albedo": np.log10(max(raw["albedo"], 1e-4)),
        "log_q": np.log10(a * (1 - e)),
        "log_moid": np.log10(raw["moid"] + 1e-6),
        "log_rms": np.log10(raw["rms"] + 1e-6),
    }
    for ang in ["om", "w", "ma"]:
        rad = np.deg2rad(raw[ang])
        f[f"{ang}_sin"], f[f"{ang}_cos"] = np.sin(rad), np.cos(rad)
    for s in ["sigma_e", "sigma_a", "sigma_i", "sigma_om", "sigma_w", "sigma_ma",
              "sigma_n", "sigma_tp", "sigma_per"]:
        f[f"log_{s}"] = np.log10(raw[s] + 1e-12)
    return pd.DataFrame([f])


def shap_waterfall(artifact, key, X):
    pipe = artifact["pipeline"]
    Xt = X
    if "scaler" in pipe.named_steps:
        Xt = pd.DataFrame(pipe.named_steps["scaler"].transform(X), columns=X.columns)
    sv = get_explainer(artifact, key)(Xt)
    if len(sv.values.shape) == 3:   # RandomForest classifier: take the PHA class
        sv = sv[:, :, 1]
    sv.data = X.values              # show original (unscaled) values on the plot
    plt.close("all")
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    return fig


# ---------- Sidebar inputs ----------
st.sidebar.title("☄️ Asteroid parameters")

PRESETS = {
    "Custom": None,
    "Main-belt asteroid": dict(H=15.0, a=2.7, e=0.15, i=10.0, om=80.0, w=150.0, ma=200.0, albedo=0.10, moid=1.7),
    "Apollo-type near-Earth object": dict(H=19.5, a=1.5, e=0.55, i=8.0, om=200.0, w=300.0, ma=40.0, albedo=0.20, moid=0.02),
    "Jupiter Trojan": dict(H=12.0, a=5.2, e=0.08, i=15.0, om=120.0, w=60.0, ma=300.0, albedo=0.07, moid=4.2),
}
preset_name = st.sidebar.selectbox("Load an example", list(PRESETS.keys()))
p = PRESETS[preset_name] or PRESETS["Main-belt asteroid"]

st.sidebar.subheader("Physical")
H = st.sidebar.number_input("Absolute magnitude H", 0.0, 35.0, p["H"], 0.1,
                            help="Brightness; lower H = brighter = usually larger")
albedo = st.sidebar.number_input("Albedo (0–1)", 0.001, 1.0, p["albedo"], 0.01,
                                 help="Surface reflectivity — used for the diameter prediction")

st.sidebar.subheader("Orbit")
a = st.sidebar.number_input("Semi-major axis a (AU)", 0.1, 500.0, p["a"], 0.05)
e = st.sidebar.number_input("Eccentricity e", 0.0, 0.99, p["e"], 0.01)
i = st.sidebar.number_input("Inclination i (deg)", 0.0, 180.0, p["i"], 0.5)
om = st.sidebar.number_input("Longitude of ascending node om (deg)", 0.0, 360.0, p["om"], 1.0)
w = st.sidebar.number_input("Argument of perihelion w (deg)", 0.0, 360.0, p["w"], 1.0)
ma = st.sidebar.number_input("Mean anomaly ma (deg)", 0.0, 360.0, p["ma"], 1.0)
moid = st.sidebar.number_input("MOID (AU)", 0.0, 100.0, p["moid"], 0.01,
                               help="Used only for diameter and clustering — NOT for hazard classification (leakage)")

defaults = {**reg_art.get("raw_defaults", {}), **clf_art.get("raw_defaults", {})}
with st.sidebar.expander("Advanced: orbit-quality values"):
    st.caption("Pre-filled with the training-data median. Change only if you know them.")
    rms = st.number_input("rms", 0.0, 10.0, float(defaults.get("rms", 0.5)), format="%.4f")
    sig = {}
    for s in ["sigma_e", "sigma_a", "sigma_i", "sigma_om", "sigma_w", "sigma_ma",
              "sigma_n", "sigma_tp", "sigma_per"]:
        sig[s] = st.number_input(s, 0.0, 1e9, float(defaults.get(s, 1e-6)), format="%.3e")

raw = dict(H=H, albedo=albedo, a=a, e=e, i=i, om=om, w=w, ma=ma, moid=moid, rms=rms, **sig)
X_full = engineer(raw)

# ---------- Main page ----------
st.title("Planetary Defense Analytics")
st.caption("Hazard classification · Diameter prediction · Orbital clustering — trained on the NASA JPL Small-Body Database")

q = a * (1 - e)
c1, c2, c3 = st.columns(3)
c1.metric("Perihelion q (AU)", f"{q:.3f}")
c2.metric("Aphelion Q (AU)", f"{a * (1 + e):.3f}")
c3.metric("Orbital period (years)", f"{a ** 1.5:.2f}")

tab1, tab2, tab3 = st.tabs(["🛡️ Hazard (PHA)", "📏 Diameter", "🪐 Cluster"])

# --- Case 1: classification
with tab1:
    X_clf = X_full[clf_art["features"]]
    prob = float(clf_art["pipeline"].predict_proba(X_clf)[0, 1])
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(clf_art.get("threshold", 0.5)), 0.05,
                          help="Lower threshold = catch more hazards (higher recall) but more false alarms")
    is_pha = prob >= threshold
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if is_pha:
            st.error("⚠️ Potentially Hazardous")
        else:
            st.success("✅ Not Potentially Hazardous")
        st.metric("PHA probability", f"{prob * 100:.1f}%")
        st.caption(f"Model: {clf_art['model_name']}")
    with col_b:
        st.markdown("**Why this prediction (SHAP)**")
        st.pyplot(shap_waterfall(clf_art, "clf", X_clf))
        st.caption("Values are in log-odds. Red bars push towards PHA, blue bars push away. MOID is deliberately not used (leakage).")

# --- Case 2: regression
with tab2:
    X_reg = X_full[reg_art["features"]]
    log_d = float(reg_art["pipeline"].predict(X_reg)[0])
    d_km = 10 ** log_d
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Predicted diameter", f"{d_km:,.3f} km" if d_km >= 1 else f"{d_km * 1000:,.0f} m")
        physics = 1329 / np.sqrt(albedo) * 10 ** (-H / 5)
        st.caption(f"Physics formula check (from H and albedo): {physics:,.3f} km")
        st.caption(f"Model: {reg_art['model_name']}")
    with col_b:
        st.markdown("**Why this prediction (SHAP, log10 diameter scale)**")
        st.pyplot(shap_waterfall(reg_art, "reg", X_reg))

# --- Case 3: clustering
with tab3:
    Xc = pd.DataFrame([{
        "log_a": np.log10(a), "e": e, "i": i,
        "log_per": np.log10(365.25 * a ** 1.5),
        "log_moid_ld": np.log10(moid * AU_TO_LD + 1e-3),
    }])[clu_art["features"]]
    cluster = int(clu_art["kmeans"].predict(clu_art["scaler"].transform(Xc))[0])
    name = clu_art["cluster_names"].get(cluster, f"Cluster {cluster}")
    st.subheader(f"Assigned to: {name}")
    st.caption(f"K-Means with k = {clu_art['k']} · silhouette = {clu_art['silhouette']:.3f}")

    sample = clu_art["plot_sample"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c in sorted(sample["cluster"].unique()):
        s = sample[sample["cluster"] == c]
        ax.scatter(s["a"], s["e"], s=4, alpha=0.4, label=f"Cluster {c}")
    ax.scatter([a], [e], marker="*", s=500, c="red", edgecolors="black", label="Your asteroid", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Semi-major axis a (AU, log scale)")
    ax.set_ylabel("Eccentricity e")
    ax.set_title("Where your asteroid sits among the catalogued population")
    leg = ax.legend(fontsize=8)
    for h in leg.legend_handles:
        h.set_sizes([40])
    st.pyplot(fig)
st.divider()
st.caption("Educational project — predictions are not an official hazard assessment. Data: NASA JPL Small-Body Database.")
