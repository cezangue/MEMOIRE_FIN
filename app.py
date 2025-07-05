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
    st.write(f"Nombre de lignes : {len(df)}")
    st.write(f"Nombre de colonnes : {len(df.columns)}")

    # Nettoyage strict des colonnes Latitude_GPS et Longitude_GPS
    if 'Latitude_GPS' in df.columns:
        df['Latitude_GPS'] = pd.to_numeric(df['Latitude_GPS'], errors='coerce')
        df = df.dropna(subset=['Latitude_GPS'])
    if 'Longitude_GPS' in df.columns:
        df['Longitude_GPS'] = pd.to_numeric(df['Longitude_GPS'], errors='coerce')
        df = df.dropna(subset=['Longitude_GPS'])

    # Vérification de la variable de pondération
    if weight_col not in df.columns:
        st.error(f"La variable de pondération '{weight_col}' n'a pas été trouvée.")
        st.write("Colonnes disponibles:", list(df.columns))
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        df = df.dropna(subset=[weight_col])
        if (df[weight_col] <= 0).any():
            st.error("La variable de pondération contient des valeurs non positives.")
        else:
            st.header("Statistiques Descriptives")
            
            # Affichage des informations sur les colonnes pour aider l'utilisateur
            st.subheader("Informations sur les colonnes disponibles")
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                col_info.append({
                    'Colonne': col,
                    'Type': dtype,
                    'Valeurs Uniques': unique_count,
                    'Valeurs Manquantes': missing_count,
                    'Pourcentage Manquant': f"{missing_pct:.1f}%"
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
            
            # Séparation automatique des colonnes par type
            all_cols = df.columns.tolist()
            
            # Identifier les colonnes numériques (excluant les colonnes système)
            numeric_cols = []
            categorical_cols = []
            
            for col in all_cols:
                if col in [weight_col, 'Latitude_GPS', 'Longitude_GPS']:
                    continue  # Ignorer les colonnes système
                
                if df[col].dtype in ['int64', 'float64']:
                    # Vérifier si c'est vraiment numérique (pas des codes catégoriels)
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.1 or df[col].nunique() > 20:  # Seuil pour considérer comme numérique
                        numeric_cols.append(col)
                    else:
                        categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            st.subheader("Sélection des Covariables")
            
            # Sélection des covariables numériques
            st.write("**Covariables Numériques Disponibles :**")
            if numeric_cols:
                quant_selected = st.multiselect(
                    "Sélectionnez les covariables numériques à utiliser pour l'appariement",
                    numeric_cols,
                    default=[],
                    help="Ces variables seront utilisées directement dans l'algorithme d'appariement"
                )
            else:
                st.write("Aucune colonne numérique détectée.")
                quant_selected = []
            
            # Sélection des covariables catégorielles
            st.write("**Covariables Catégorielles Disponibles :**")
            if categorical_cols:
                qual_selected = st.multiselect(
                    "Sélectionnez les covariables catégorielles à transformer en variables dummy",
                    categorical_cols,
                    default=[],
                    help="Ces variables seront converties en variables binaires (0/1) avant l'appariement"
                )
                
                # Affichage des modalités pour les variables catégorielles sélectionnées
                if qual_selected:
                    st.write("**Modalités des variables catégorielles sélectionnées :**")
                    for var in qual_selected:
                        modalites = df[var].value_counts().head(10)
                        st.write(f"- {var}: {list(modalites.index)}")
                        if len(modalites) > 10:
                            st.write(f"  ... et {df[var].nunique() - 10} autres modalités")
            else:
                st.write("Aucune colonne catégorielle détectée.")
                qual_selected = []

            # Vérification que l'utilisateur a sélectionné au moins une covariable
            if not quant_selected and not qual_selected:
                st.warning("Veuillez sélectionner au moins une covariable pour procéder à l'analyse.")
            else:
                # Préparation des données avec dummification
                X_cols = quant_selected.copy()
                df_working = df.copy()
                
                # Traitement des variables catégorielles sélectionnées
                if qual_selected:
                    st.write("**Création des variables dummy...**")
                    for var in qual_selected:
                        # Affichage du nombre de modalités
                        n_modalites = df_working[var].nunique()
                        st.write(f"- {var}: {n_modalites} modalités → {n_modalites-1} variables dummy")
                    
                    # Création des variables dummy
                    df_dummied = pd.get_dummies(df_working[qual_selected], drop_first=True, prefix=qual_selected)
                    df_working = pd.concat([df_working, df_dummied], axis=1)
                    X_cols.extend(df_dummied.columns.tolist())
                
                st.write(f"**Variables sélectionnées pour l'appariement ({len(X_cols)} au total) :**")
                st.write(X_cols)
                
                # Vérification de la qualité des données pour les variables sélectionnées
                st.subheader("Qualité des données pour les variables sélectionnées")
                quality_check = []
                for col in X_cols:
                    if col in df_working.columns:
                        missing_count = df_working[col].isnull().sum()
                        missing_pct = (missing_count / len(df_working)) * 100
                        if col in quant_selected:
                            # Vérifier si la conversion numérique est possible
                            try:
                                temp_series = pd.to_numeric(df_working[col], errors='coerce')
                                conversion_loss = temp_series.isnull().sum() - df_working[col].isnull().sum()
                                quality_check.append({
                                    'Variable': col,
                                    'Type': 'Numérique',
                                    'Valeurs Manquantes': missing_count,
                                    'Pourcentage Manquant': f"{missing_pct:.1f}%",
                                    'Perte Conversion': conversion_loss,
                                    'Statut': 'OK' if conversion_loss == 0 else 'Attention'
                                })
                            except:
                                quality_check.append({
                                    'Variable': col,
                                    'Type': 'Numérique',
                                    'Valeurs Manquantes': missing_count,
                                    'Pourcentage Manquant': f"{missing_pct:.1f}%",
                                    'Perte Conversion': 'Erreur',
                                    'Statut': 'Problème'
                                })
                        else:
                            quality_check.append({
                                'Variable': col,
                                'Type': 'Dummy',
                                'Valeurs Manquantes': missing_count,
                                'Pourcentage Manquant': f"{missing_pct:.1f}%",
                                'Perte Conversion': 'N/A',
                                'Statut': 'OK'
                            })
                
                quality_df = pd.DataFrame(quality_check)
                st.dataframe(quality_df, use_container_width=True)
                
                # Avertissement si des problèmes sont détectés
                if 'Problème' in quality_df['Statut'].values:
                    st.error("⚠️ Certaines variables présentent des problèmes de conversion. Vérifiez les données.")
                elif 'Attention' in quality_df['Statut'].values:
                    st.warning("⚠️ Certaines variables perdront des observations lors de la conversion numérique.")

                st.header("Visualisation de la Population")
                geo_level = st.selectbox("Niveau Géographique", 
                                       ['Region', 'Departement', 'Commune'] if any(col in df.columns for col in ['Region', 'Departement', 'Commune']) else [])
                if geo_level and geo_level in df.columns:
                    pop_table = df.groupby(geo_level).agg({weight_col: 'sum'}).rename(columns={weight_col: 'Poids Total'})
                    st.table(pop_table)
                    if 'Latitude_GPS' in df.columns and 'Longitude_GPS' in df.columns:
                        if df['Latitude_GPS'].dtype in ['float64', 'int64'] and df['Longitude_GPS'].dtype in ['float64', 'int64']:
                            fig = px.scatter_mapbox(df, lat='Latitude_GPS', lon='Longitude_GPS', 
                                                 color='EAU', zoom=5, title="Répartition géographique")
                            st.plotly_chart(fig)
                        else:
                            st.warning("Les colonnes Latitude_GPS ou Longitude_GPS contiennent des données non numériques.")

                st.header("Évaluation d'Impact")
                method = st.selectbox("Méthode", ["Double Différence"])
                
                if method == "Double Différence":
                    # Vérification des colonnes nécessaires
                    required_cols = ['durée_disponibilite_acces_eau_avantH', 'durée_dispisponibilite_acces_eau_apresH', 'EAU']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"Colonnes manquantes pour l'analyse : {missing_cols}")
                    else:
                        # Préparation des données pour l'analyse
                        analysis_df = df_working.copy()
                        
                        # Nettoyage des variables de résultat
                        analysis_df['durée_disponibilite_acces_eau_avantH'] = pd.to_numeric(
                            analysis_df['durée_disponibilite_acces_eau_avantH'], errors='coerce')
                        analysis_df['durée_dispisponibilite_acces_eau_apresH'] = pd.to_numeric(
                            analysis_df['durée_dispisponibilite_acces_eau_apresH'], errors='coerce')
                        
                        # Suppression des observations avec des valeurs manquantes sur les variables clés
                        key_vars = ['durée_disponibilite_acces_eau_avantH', 'durée_dispisponibilite_acces_eau_apresH', 'EAU', weight_col] + X_cols
                        analysis_df = analysis_df.dropna(subset=key_vars)
                        
                        # Nettoyage des covariables numériques
                        for col in quant_selected:
                            if col in analysis_df.columns:
                                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
                        
                        # Suppression finale des valeurs manquantes
                        analysis_df = analysis_df.dropna(subset=X_cols)
                        
                        st.write(f"**Données après nettoyage : {len(analysis_df)} observations**")
                        
                        if len(analysis_df) == 0:
                            st.error("Aucune donnée valide après nettoyage. Vérifiez les variables sélectionnées.")
                        else:
                            # Définition des groupes
                            Y_before = analysis_df['durée_disponibilite_acces_eau_avantH']
                            Y_after = analysis_df['durée_dispisponibilite_acces_eau_apresH']
                            T = analysis_df['EAU'] == 'Non'  # Groupe de traitement
                            weights = analysis_df[weight_col]
                            
                            benef = analysis_df[T].copy()
                            non_benef = analysis_df[~T].copy()
                            
                            st.write(f"**Répartition des groupes :**")
                            st.write(f"- Bénéficiaires (EAU = 'Non'): {len(benef)} observations")
                            st.write(f"- Non-bénéficiaires (EAU ≠ 'Non'): {len(non_benef)} observations")
                            st.write(f"- Valeurs uniques de EAU: {analysis_df['EAU'].unique()}")
                            
                            if len(benef) == 0 or len(non_benef) == 0:
                                st.error("Un des groupes est vide. Vérifiez la variable de traitement 'EAU'.")
                            else:
                                # Extraction des matrices de covariables
                                X_benef = benef[X_cols].values
                                X_non_benef = non_benef[X_cols].values
                                
                                st.write(f"**Matrices de covariables :**")
                                st.write(f"- X_benef: {X_benef.shape}")
                                st.write(f"- X_non_benef: {X_non_benef.shape}")
                                
                                # Processus d'appariement par région
                                if 'Region' in analysis_df.columns:
                                    regions = analysis_df['Region'].unique()
                                    st.write(f"**Appariement par région ({len(regions)} régions) :**")
                                    
                                    matches = []
                                    for region in regions:
                                        benef_region = benef[benef['Region'] == region].copy()
                                        non_benef_region = non_benef[non_benef['Region'] == region].copy()
                                        
                                        if len(benef_region) > 0 and len(non_benef_region) > 0:
                                            X_benef_region = benef_region[X_cols].values
                                            X_non_benef_region = non_benef_region[X_cols].values
                                            
                                            if X_benef_region.size > 0 and X_non_benef_region.size > 0:
                                                try:
                                                    # Calcul de la distance de Mahalanobis
                                                    combined_data = np.vstack([X_benef_region, X_non_benef_region])
                                                    cov_matrix = np.cov(combined_data.T)
                                                    cov_inv = pinv(cov_matrix)
                                                    
                                                    distances = cdist(X_non_benef_region, X_benef_region, 
                                                                    metric='mahalanobis', VI=cov_inv)
                                                    threshold = np.percentile(distances, 90)
                                                    
                                                    region_matches = 0
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
                                                                'Region': region,
                                                                'distance': distances[i, min_dist_idx]
                                                            })
                                                            region_matches += 1
                                                    
                                                    st.write(f"  - {region}: {region_matches} appariements sur {len(non_benef_region)} non-bénéficiaires")
                                                    
                                                except Exception as e:
                                                    st.warning(f"Erreur dans l'appariement pour la région {region}: {str(e)}")
                                        else:
                                            st.write(f"  - {region}: données insuffisantes (benef: {len(benef_region)}, non-benef: {len(non_benef_region)})")
                                    
                                    # Calcul des résultats
                                    if matches:
                                        matched_df = pd.DataFrame(matches)
                                        st.write(f"**Total des appariements réussis : {len(matched_df)}**")
                                        
                                        # Ajustement des poids
                                        matched_df['benef_weight'] = matched_df.groupby('benef_id')['weight'].transform('sum')
                                        matched_df['adjusted_weight'] = matched_df['weight'] / matched_df['benef_weight']
                                        
                                        # Calcul des résultats par région et au niveau national
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
                                        
                                        # Résultat national
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
                                        
                                        st.subheader("Résultats de l'Analyse Double Différence")
                                        results_df = pd.DataFrame(data)
                                        st.dataframe(results_df, use_container_width=True)
                                        
                                        # Statistiques d'appariement
                                        st.subheader("Qualité de l'Appariement")
                                        st.write(f"- Distance moyenne: {matched_df['distance'].mean():.4f}")
                                        st.write(f"- Distance médiane: {matched_df['distance'].median():.4f}")
                                        st.write(f"- Distance max: {matched_df['distance'].max():.4f}")
                                        
                                    else:
                                        st.error("Aucun appariement réussi. Vérifiez les covariables sélectionnées et les données.")
                                else:
                                    st.error("La colonne 'Region' n'est pas disponible pour l'appariement par région.")
