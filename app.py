import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, chi2_contingency, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from numpy.linalg import pinv
from sklearn.ensemble import GradientBoostingRegressor
import io
import warnings
import uuid
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Plateforme d'Analyse d'Impact - MINEPAT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne et professionnel
st.markdown("""
<style>
    .main-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .section-title {
        background: linear-gradient(90deg, #e9ecef 0%, #dee2e6 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 8px;
        color: white;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="header">
    <h1>🏛️ Plateforme d'Analyse d'Impact - MINEPAT</h1>
    <h3>Ministère de l'Économie et de la Planification du Territoire</h3>
    <p>Analyse Avancée des Politiques d'Accès à l'Eau Potable</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour la configuration
with st.sidebar:
    st.markdown("### 🎛️ Configuration")
    uploaded_file = st.file_uploader(
        "📂 Importer les données",
        type=["xlsx", "csv"],
        help="Formats supportés : Excel (.xlsx) ou CSV (.csv)"
    )
    weight_col = st.text_input(
        "Variable de pondération",
        value="Ponderation_Strate_region",
        help="Colonne contenant les poids pour l'analyse"
    )
    filter_zone = st.selectbox(
        "Filtre Zone Écologique",
        ["Tous"] + [f"Zone {i}" for i in range(1, 6)],
        help="Filtrer les données par zone écologique"
    )
    theme = st.selectbox(
        "Thème de visualisation",
        ["Plotly", "Seaborn", "Streamlit"],
        help="Choisir le style des visualisations"
    )

# Fonction pour créer des cartes de métriques
def create_metric_card(title, value, delta=None, delta_color="normal"):
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
            {"<p style='color:" + ("green" if delta_color == "normal" else "red") + f";'>{delta}</p>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# Fonction pour visualisations avancées
def create_advanced_visualization(df, variable, title, theme="Plotly"):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution", "Box Plot"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    fig.add_trace(go.Histogram(x=df[variable], name="Distribution", opacity=0.75), row=1, col=1)
    fig.add_trace(go.Box(y=df[variable], name="Box Plot"), row=1, col=2)
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        height=400,
        template=theme.lower()
    )
    return fig

# Traitement principal
if uploaded_file is not None:
    try:
        # Chargement des données
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Application du filtre de zone
        if filter_zone != "Tous" and 'Zone ecologique' in df.columns:
            df = df[df['Zone ecologique'] == filter_zone]
        
        # Onglets de navigation
        tabs = st.tabs([
            "📊 Données",
            "🔍 Exploration",
            "🎯 Variables",
            "📈 Impact",
            "📋 Rapport"
        ])
        
        with tabs[0]:
            st.markdown('<div class="section-title"><h2>📊 Qualité des Données</h2></div>', unsafe_allow_html=True)
            
            # Métriques clés
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                create_metric_card("Observations", f"{len(df):,}")
            with col2:
                create_metric_card("Variables", f"{len(df.columns):,}")
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                create_metric_card("Valeurs Manquantes", f"{missing_pct:.1f}%")
            with col4:
                create_metric_card("Régions", f"{df['Region'].nunique()}" if 'Region' in df.columns else "N/A")
            
            # Aperçu des données
            st.markdown("### 📋 Aperçu des Données")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Analyse de qualité
            st.markdown("### 🔍 Rapport de Qualité")
            col_info = [
                {
                    'Variable': col,
                    'Type': 'Numérique' if df[col].dtype in ['int64', 'float64'] else 'Catégorielle',
                    'Uniques': df[col].nunique(),
                    'Manquants': df[col].isnull().sum(),
                    '% Manquant': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                    'Statut': '✅ Bon' if (df[col].isnull().sum() / len(df)) * 100 < 10 else '⚠️ Attention'
                }
                for col in df.columns
            ]
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        with tabs[1]:
            st.markdown('<div class="section-title"><h2>🔍 Analyse Exploratoire</h2></div>', unsafe_allow_html=True)
            
            if weight_col not in df.columns:
                st.markdown(f'<div class="error-box">❌ Variable de pondération "{weight_col}" introuvable</div>', unsafe_allow_html=True)
            else:
                df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
                df = df.dropna(subset=[weight_col])
                
                if (df[weight_col] <= 0).any():
                    st.markdown('<div class="error-box">❌ Valeurs non positives dans la variable de pondération</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✅ Variable de pondération validée</div>', unsafe_allow_html=True)
                    
                    # Carte interactive
                    if 'Latitude_GPS' in df.columns and 'Longitude_GPS' in df.columns:
                        st.markdown("### 🗺️ Visualisation Géographique")
                        fig = px.scatter_mapbox(
                            df,
                            lat='Latitude_GPS',
                            lon='Longitude_GPS',
                            color='EAU' if 'EAU' in df.columns else None,
                            size=weight_col,
                            zoom=5,
                            mapbox_style="open-street-map",
                            title="Distribution Géographique des Interventions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown('<div class="section-title"><h2>🎯 Sélection des Variables</h2></div>', unsafe_allow_html=True)
            
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in [weight_col, 'Latitude_GPS', 'Longitude_GPS']]
            categorical_cols = [col for col in df.columns if col not in numeric_cols + [weight_col, 'Latitude_GPS', 'Longitude_GPS']]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔢 Variables Numériques")
                quant_selected = st.multiselect("Variables numériques", numeric_cols, default=[])
                for col in quant_selected[:3]:
                    with st.expander(f"📊 {col}"):
                        st.plotly_chart(create_advanced_visualization(df, col, f"Analyse de {col}", theme))
            
            with col2:
                st.markdown("### 🏷️ Variables Catégorielles")
                qual_selected = st.multiselect("Variables catégorielles", categorical_cols, default=[])
                for col in qual_selected[:3]:
                    with st.expander(f"📝 {col}"):
                        fig = px.pie(names=df[col].value_counts().index, values=df[col].value_counts().values, title=f"Répartition de {col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            if quant_selected or qual_selected:
                df_working = df.copy()
                X_cols = quant_selected.copy()
                if qual_selected:
                    df_dummied = pd.get_dummies(df_working[qual_selected], drop_first=True)
                    df_working = pd.concat([df_working, df_dummied], axis=1)
                    X_cols.extend(df_dummied.columns.tolist())
                st.session_state.X_cols = X_cols
                st.session_state.df_working = df_working
        
        with tabs[3]:
            st.markdown('<div class="section-title"><h2>📈 Analyse d'Impact</h2></div>', unsafe_allow_html=True)
            
            if 'X_cols' not in st.session_state:
                st.markdown('<div class="warning-box">⚠️ Sélectionnez d\'abord des variables</div>', unsafe_allow_html=True)
            else:
                method = st.selectbox("Méthode d'analyse", ["Double Différence", "Score de Propension"])
                
                if method == "Double Différence":
                    required_cols = ['durée_disponibilite_acces_eau_avantH', 'durée_dispisponibilite_acces_eau_apresH', 'EAU']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.markdown(f'<div class="error-box">❌ Colonnes manquantes : {missing_cols}</div>', unsafe_allow_html=True)
                    else:
                        analysis_df = st.session_state.df_working.copy()
                        analysis_df = analysis_df.dropna(subset=required_cols + st.session_state.X_cols)
                        
                        benef = analysis_df[analysis_df['EAU'] == 'Non'].copy()
                        non_benef = analysis_df[analysis_df['EAU'] != 'Non'].copy()
                        
                        if 'Region' in analysis_df.columns:
                            regions = analysis_df['Region'].unique()
                            matches = []
                            for region in regions:
                                benef_region = benef[benef['Region'] == region]
                                non_benef_region = non_benef[non_benef['Region'] == region]
                                
                                if len(benef_region) > 0 and len(non_benef_region) > 0:
                                    X_benef = benef_region[st.session_state.X_cols].values
                                    X_non_benef = non_benef_region[st.session_state.X_cols].values
                                    
                                    cov_matrix = np.cov(np.vstack([X_benef, X_non_benef]).T)
                                    cov_inv = pinv(cov_matrix)
                                    distances = cdist(X_non_benef, X_benef, metric='mahalanobis', VI=cov_inv)
                                    
                                    for i in range(len(non_benef_region)):
                                        min_dist_idx = np.argmin(distances[i])
                                        matches.append({
                                            'non_benef_id': non_benef_region.index[i],
                                            'benef_id': benef_region.index[min_dist_idx],
                                            'non_benef_avant': non_benef_region.iloc[i]['durée_disponibilite_acces_eau_avantH'],
                                            'non_benef_apres': non_benef_region.iloc[i]['durée_dispisponibilite_acces_eau_apresH'],
                                            'benef_avant': benef_region.iloc[min_dist_idx]['durée_disponibilite_acces_eau_avantH'],
                                            'benef_apres': benef_region.iloc[min_dist_idx]['durée_dispisponibilite_acces_eau_apresH'],
                                            'weight': non_benef_region.iloc[i][weight_col],
                                            'Region': region
                                        })
                            
                            if matches:
                                matched_df = pd.DataFrame(matches)
                                results = []
                                for region in regions:
                                    df_region = matched_df[matched_df['Region'] == region]
                                    if not df_region.empty:
                                        benef_diff = df_region['benef_apres'].mean() - df_region['benef_avant'].mean()
                                        non_benef_diff = df_region['non_benef_apres'].mean() - df_region['non_benef_avant'].mean()
                                        results.append({
                                            'Région': region,
                                            'Effet (DiD)': benef_diff - non_benef_diff,
                                            'Effectif': len(df_region)
                                        })
                                st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        with tabs[4]:
            st.markdown('<div class="section-title"><h2>📋 Rapport Final</h2></div>', unsafe_allow_html=True)
            st.write("Génération du rapport final en cours...")
            # Placeholder pour rapport détaillé

    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Erreur : {str(e)}</div>', unsafe_allow_html=True)
