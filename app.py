import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from numpy.linalg import pinv, inv
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import io
import warnings
warnings.filterwarnings("ignore")

st.title("Tableau de Bord d'Évaluation de l'Impact de l'Accès à l'Eau")

st.header("Télécharger les Données")
uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])
weight_col = st.text_input("Entrez la variable de pondération", "Ponderation_Strate_region")
filter_zone = st.selectbox("Filtrer par Zone Écologique", ["Tous"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if filter_zone != "Tous" and 'Zone ecologique' in df.columns:
        df = df[df['Zone ecologique'] == filter_zone]
    st.write("Aperçu des Données:", df.head())

    # Nettoyage strict des colonnes Latitude_GPS et Longitude_GPS
    if 'Latitude_GPS' in df.columns:
        df['Latitude_GPS'] = pd.to_numeric(df['Latitude_GPS'], errors='coerce')
        df = df.dropna(subset=['Latitude_GPS'])
    if 'Longitude_GPS' in df.columns:
        df['Longitude_GPS'] = pd.to_numeric(df['Longitude_GPS'], errors='coerce')
        df = df.dropna(subset=['Longitude_GPS'])

    if weight_col not in df.columns:
        st.error(f"La variable de pondération '{weight_col}' n'a pas été trouvée.")
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        df = df.dropna(subset=[weight_col])
        if (df[weight_col] <= 0).any():
            st.error("La variable de pondération contient des valeurs non positives.")
        else:
            st.header("Statistiques Descriptives")
            quant_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            qual_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            quant_selected = st.multiselect("Variables Quantitatives", quant_cols, default=quant_cols[:3])
            if quant_selected:
                stats = []
                for col in quant_selected:
                    mean = np.average(df[col].dropna(), weights=df[weight_col].loc[df[col].notna()])
                    median = np.percentile(df[col].dropna(), 50)
                    stats.append({'Variable': col, 'Moyenne': round(mean, 2), 'Médiane': round(median, 2)})
                st.table(pd.DataFrame(stats))
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=quant_selected[0], weights=df[weight_col], kde=True, ax=ax)
                st.pyplot(fig)

            qual_selected = st.multiselect("Variables Qualitatives", qual_cols, default=qual_cols[:3])
            if qual_selected:
                for col in qual_selected:
                    freq = df.groupby(col)[weight_col].sum()
                    perc = (freq / freq.sum() * 100).round(2)
                    st.table(pd.DataFrame({'Fréquence': freq, 'Pourcentage': perc}))

            st.header("Visualisation de la Population")
            geo_level = st.selectbox("Niveau Géographique", ['Region', 'Departement', 'Commune'] if any(col in df.columns for col in ['Region', 'Departement', 'Commune']) else [])
            if geo_level and geo_level in df.columns:
                pop_table = df.groupby(geo_level).agg({weight_col: 'sum'}).rename(columns={weight_col: 'Poids Total'})
                st.table(pop_table)
                if 'Latitude_GPS' in df.columns and 'Longitude_GPS' in df.columns:
                    # Vérifier que les colonnes sont numériques avant la visualisation
                    if df['Latitude_GPS'].dtype in ['float64', 'int64'] and df['Longitude_GPS'].dtype in ['float64', 'int64']:
                        fig = px.scatter_mapbox(df, lat='Latitude_GPS', lon='Longitude_GPS', color='EAU', zoom=5)
                        st.plotly_chart(fig)
                    else:
                        st.warning("Les colonnes Latitude_GPS ou Longitude_GPS contiennent des données non numériques.")

            st.header("Évaluation d'Impact")
            method = st.selectbox("Méthode", ["Double Différence"])
            if method == "Double Différence":
                if 'durée_disponibilite_acces_eau_avantH' in df.columns and 'durée_dispisponibilite_acces_eau_apresH' in df.columns:
                    Y_before = df['durée_disponibilite_acces_eau_avantH']
                    Y_after = df['durée_dispisponibilite_acces_eau_apresH']
                    T = df['EAU'] == 'Non'
                    weights = df[weight_col]
                    benef = df[T].copy()
                    non_benef = df[~T].copy()

                    # Convertir et valider les colonnes pour X_benef et X_non_benef
                    numeric_cols = benef.select_dtypes(include=['int64', 'float64']).columns
                    X_cols = [col for col in benef.columns if col not in ['durée_disponibilite_acces_eau_avantH', 'durée_dispisponibilite_acces_eau_apresH', weight_col, 'EAU']]
                    X_cols = [col for col in X_cols if col in numeric_cols]
                    if not X_cols:
                        st.error("Aucune colonne numérique disponible pour l'évaluation d'impact.")
                    else:
                        benef[X_cols] = benef[X_cols].apply(pd.to_numeric, errors='coerce')
                        non_benef[X_cols] = non_benef[X_cols].apply(pd.to_numeric, errors='coerce')
                        benef = benef.dropna(subset=X_cols)
                        non_benef = non_benef.dropna(subset=X_cols)

                        X_benef = benef[X_cols].values
                        X_non_benef = non_benef[X_cols].values

                        if X_benef.size == 0 or X_non_benef.size == 0:
                            st.warning("Aucune donnée numérique valide pour le matching.")
                        else:
                            regions = df['Region'].unique()
                            matches = []
                            for region in regions:
                                benef_region = benef[benef['Region'] == region]
                                non_benef_region = non_benef[non_benef['Region'] == region]
                                if len(benef_region) > 0 and len(non_benef_region) > 0:
                                    X_benef_region = benef_region[X_cols].values
                                    X_non_benef_region = non_benef_region[X_cols].values
                                    if X_benef_region.size > 0 and X_non_benef_region.size > 0:
                                        cov_matrix = np.cov(np.vstack([X_benef_region, X_non_benef_region]).T)
                                        cov_inv = pinv(cov_matrix)
                                        distances = cdist(X_non_benef_region, X_benef_region, metric='mahalanobis', VI=cov_inv)
                                        threshold = np.percentile(distances, 90)
                                        for i in range(len(non_benef_region)):
                                            min_dist_idx = np.argmin(distances[i])
                                            if distances[i, min_dist_idx] < threshold:
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
                            matched_df = pd.DataFrame(matches)
                            if not matched_df.empty:
                                matched_df['benef_weight'] = matched_df.groupby('benef_id')['weight'].transform('sum')
                                matched_df['adjusted_weight'] = matched_df['weight'] / matched_df['benef_weight']
                                data = []
                                for region in regions:
                                    df_region = matched_df[matched_df['Region'] == region]
                                    if not df_region.empty:
                                        benef_before = np.average(df_region['benef_avant'], weights=df_region['adjusted_weight'])
                                        benef_after = np.average(df_region['benef_apres'], weights=df_region['adjusted_weight'])
                                        non_benef_before = np.average(df_region['non_benef_avant'], weights=df_region['weight'])
                                        non_benef_after = np.average(df_region['non_benef_apres'], weights=df_region['weight'])
                                        benef_diff = benef_after - benef_before
                                        non_benef_diff = non_benef_after - non_benef_before
                                        did = benef_diff - non_benef_diff
                                        data.append({
                                            'Région': region,
                                            'Bénéficiaires Avant': round(benef_before, 2),
                                            'Bénéficiaires Après': round(benef_after, 2),
                                            'Bénéficiaires Diff': round(benef_diff, 2),
                                            'Non-bénéficiaires Avant': round(non_benef_before, 2),
                                            'Non-bénéficiaires Après': round(non_benef_after, 2),
                                            'Non-bénéficiaires Diff': round(non_benef_diff, 2),
                                            'DiD': round(did, 2),
                                            'Effectif Apparié': len(df_region),
                                            'Poids Total': round(df_region['weight'].sum(), 2)
                                        })
                                benef_before = np.average(matched_df['benef_avant'], weights=matched_df['adjusted_weight'])
                                benef_after = np.average(matched_df['benef_apres'], weights=matched_df['adjusted_weight'])
                                non_benef_before = np.average(matched_df['non_benef_avant'], weights=matched_df['weight'])
                                non_benef_after = np.average(matched_df['non_benef_apres'], weights=matched_df['weight'])
                                benef_diff = benef_after - benef_before
                                non_benef_diff = non_benef_after - non_benef_before
                                did = benef_diff - non_benef_diff
                                data.append({
                                    'Région': 'Total National',
                                    'Bénéficiaires Avant': round(benef_before, 2),
                                    'Bénéficiaires Après': round(benef_after, 2),
                                    'Bénéficiaires Diff': round(benef_diff, 2),
                                    'Non-bénéficiaires Avant': round(non_benef_before, 2),
                                    'Non-bénéficiaires Après': round(non_benef_after, 2),
                                    'Non-bénéficiaires Diff': round(non_benef_diff, 2),
                                    'DiD': round(did, 2),
                                    'Effectif Apparié': len(matched_df),
                                    'Poids Total': round(matched_df['weight'].sum(), 2)
                                })
                                st.table(pd.DataFrame(data))
