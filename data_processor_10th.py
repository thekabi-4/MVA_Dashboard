# data_processor_10th.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import os
import json
import joblib
from sqlalchemy import create_engine, text
import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
import hashlib
from datetime import datetime

# Academic columns to include in analysis
ACADEMIC_COLS_SUFFIX = '_Total'
METADATA_COLS = {'Sr_No', 'Roll_No', 'Name', 'New_Old', 'Cluster'}

def generate_academic_data_hash(df):
    """Generate hash based only on academic score columns."""
    academic_cols = [col for col in df.columns if col.endswith(ACADEMIC_COLS_SUFFIX) or col == 'Percentage']
    if not academic_cols:
        academic_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in METADATA_COLS]
    df_academic = df[academic_cols].copy().sort_index(axis=1).fillna('NaN')
    df_str = df_academic.to_csv(index=False).encode('utf-8')
    return hashlib.md5(df_str).hexdigest()

def load_and_parse_10th_sheet(file_path):
    try:
        df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
        header_row_index = None
        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            if len(row) >= 3 and str(row.iloc[0]).strip() == 'Sr.No.' and \
                               str(row.iloc[1]).strip() == 'Roll No.' and \
                               str(row.iloc[2]).strip() == 'Name':
                header_row_index = i
                break
        if header_row_index is None:
            print("Header row not found.")
            return pd.DataFrame(), {}

        students_data = []
        for i in range(header_row_index + 1, len(df_raw)):
            row = df_raw.iloc[i]
            if row.count() < 5:
                break
            try:
                sr_no = pd.to_numeric(row.iloc[0], errors='coerce')
                name = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else None
                if pd.isna(sr_no) or not name:
                    continue
                student = {
                    'Sr_No': sr_no,
                    'Roll_No': str(row.iloc[1]) if len(row) > 1 else None,
                    'Name': name.title(),
                    'English_Theory': pd.to_numeric(row.iloc[3], errors='coerce'),
                    'English_Practical': pd.to_numeric(row.iloc[4], errors='coerce'),
                    'English_Total': pd.to_numeric(row.iloc[5], errors='coerce'),
                    'Hindi_Theory': pd.to_numeric(row.iloc[6], errors='coerce'),
                    'Hindi_Practical': pd.to_numeric(row.iloc[7], errors='coerce'),
                    'Hindi_Total': pd.to_numeric(row.iloc[8], errors='coerce'),
                    'Maths_Theory': pd.to_numeric(row.iloc[9], errors='coerce'),
                    'Maths_Practical': pd.to_numeric(row.iloc[10], errors='coerce'),
                    'Maths_Total': pd.to_numeric(row.iloc[11], errors='coerce'),
                    'Science_Theory': pd.to_numeric(row.iloc[12], errors='coerce'),
                    'Science_Practical': pd.to_numeric(row.iloc[13], errors='coerce'),
                    'Science_Total': pd.to_numeric(row.iloc[14], errors='coerce'),
                    'Social_Science_Theory': pd.to_numeric(row.iloc[15], errors='coerce'),
                    'Social_Science_Practical': pd.to_numeric(row.iloc[16], errors='coerce'),
                    'Social_Science_Total': pd.to_numeric(row.iloc[17], errors='coerce'),
                    'AI_Theory': pd.to_numeric(row.iloc[18], errors='coerce'),
                    'AI_Practical': pd.to_numeric(row.iloc[19], errors='coerce'),
                    'AI_Total': pd.to_numeric(row.iloc[20], errors='coerce'),
                    'Grand_Total': pd.to_numeric(row.iloc[21], errors='coerce'),
                    'Percentage': pd.to_numeric(row.iloc[22], errors='coerce'),
                    'New_Old': None
                }
                for idx in range(23, min(30, len(row))):
                    val = str(row.iloc[idx]).strip().upper()
                    if val in ['NEW', 'OLD']:
                        student['New_Old'] = val
                        break
                students_data.append(student)
            except Exception as e:
                print(f"Error parsing row {i}: {e}")
        df = pd.DataFrame(students_data)
        return df, {}
    except Exception as e:
        print(f"File processing error: {e}")
        return pd.DataFrame(), {}

def perform_advanced_analysis(df):
    results = {}
    models = {}
    if df.empty:
        return results, models

    # Identify subject totals
    subject_totals = [col for col in df.columns if col.endswith('_Total')]
    academic_df = df[[col for col in df.columns if col not in METADATA_COLS]]

    # --- 1. Basic Stats ---
    results['basic_stats'] = {
        'total_students': len(df),
        'avg_percentage': float(df['Percentage'].mean()) if 'Percentage' in df else None,
        'std_percentage': float(df['Percentage'].std()) if 'Percentage' in df else None,
        'median_percentage': float(df['Percentage'].median()) if 'Percentage' in df else None,
        'subject_means': df[subject_totals].mean().to_dict() if subject_totals else {}
    }

    # --- 2. New/Old Comparison ---
    if 'New_Old' in df.columns and df['New_Old'].nunique() > 1:
        results['comparison_stats'] = df.groupby('New_Old')['Percentage'].describe().to_dict()
        groups = df['New_Old'].unique()
        if len(groups) == 2:
            g1 = df[df['New_Old'] == groups[0]]['Percentage'].dropna()
            g2 = df[df['New_Old'] == groups[1]]['Percentage'].dropna()
            if len(g1) > 1 and len(g2) > 1:
                t_stat, p_val = stats.ttest_ind(g1, g2)
                results['t_test'] = {'statistic': float(t_stat), 'p_value': float(p_val)}

    # --- 3. Correlation ---
    numeric_cols = academic_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr = academic_df[numeric_cols].corr()
        results['correlation_matrix'] = corr.round(4).to_dict()

    # --- 4. Outlier Detection ---
    if 'Percentage' in df.columns:
        Q1 = df['Percentage'].quantile(0.25)
        Q3 = df['Percentage'].quantile(0.75)
        IQR = Q3 - Q1
        bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        iqr_outliers = df[(df['Percentage'] < bounds[0]) | (df['Percentage'] > bounds[1])]
        results['outliers_iqr'] = iqr_outliers[['Name', 'Percentage']].to_dict('records')

        # Z-Score
        z_scores = np.abs(stats.zscore(df['Percentage'].dropna()))
        z_outliers = df.iloc[df['Percentage'].dropna().index][(z_scores > 3)]
        results['outliers_zscore'] = z_outliers[['Name', 'Percentage']].to_dict('records')

        # Isolation Forest
        iso_features = [col for col in subject_totals + ['Grand_Total', 'Percentage'] if col in df.columns]
        if iso_features:
            df_iso = df[iso_features].dropna()
            if len(df_iso) > 10:
                iso = IsolationForest(contamination=0.1, random_state=42)
                preds = iso.fit_predict(df_iso)
                outlier_idx = df_iso.index[preds == -1]
                results['outliers_isolation_forest'] = df.loc[outlier_idx, ['Name'] + iso_features].to_dict('records')
                results['features_used_for_outlier_detection'] = iso_features
                models['isolation_forest'] = iso

    # --- 5. Predictive Modeling ---
    if subject_totals and 'Percentage' in df.columns:
        data = df[subject_totals + ['Percentage']].dropna()
        if len(data) > 10:
            X, y = data[subject_totals], data['Percentage']
            lr = LinearRegression()
            lr.fit(X, y)
            results['linear_regression_r2'] = float(lr.score(X, y))
            results['linear_regression_coefficients'] = dict(zip(subject_totals, lr.coef_.round(4).tolist()))
            results['feature_names_for_modeling'] = subject_totals
            models['linear_regression'] = lr

    # --- 6. Clustering ---
    if subject_totals:
        df_clust = df[subject_totals].dropna()
        if len(df_clust) > 5:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clust)
            models['scaler'] = scaler
            best_k, best_sil = 2, -1
            sil_scores = {}
            for k in range(2, min(6, len(df_clust))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels) if k > 1 else -1
                sil_scores[k] = sil
                if sil > best_sil:
                    best_sil, best_k = sil, k
            kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels_final = kmeans_final.fit_predict(X_scaled)
            df.loc[df_clust.index, 'Cluster'] = labels_final
            cluster_summary = df.groupby('Cluster')[subject_totals + (['Percentage'] if 'Percentage' in df else [])].mean().round(2)
            results['clustering'] = {
                'optimal_k': best_k,
                'silhouette_scores': sil_scores,
                'cluster_summary': cluster_summary.to_dict()
            }
            models['kmeans'] = kmeans_final

    # --- 7. PCA ---
    if len(subject_totals) > 2:
        df_pca = df[subject_totals].dropna()
        if len(df_pca) > 5:
            scaler = StandardScaler()
            X_pca = scaler.fit_transform(df_pca)
            pca = PCA(n_components=2)
            comps = pca.fit_transform(X_pca)
            df_pca = df_pca.copy()
            df_pca['PC1'] = comps[:, 0]
            df_pca['PC2'] = comps[:, 1]
            results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist()
            }
            results['pca_data'] = df_pca[['PC1', 'PC2']].reset_index(drop=True).to_dict('records')

    results['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return results, models

def process_and_save_to_sql(df, results, models, db_config, data_hash=None):
    """
    Saves analysis results to MySQL. Creates DB and tables if they don't exist.
    """
    try:
        # Step 1: Connect to MySQL server (without specifying DB) to create DB if needed
        server_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}"
        server_engine = create_engine(server_url)

        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_config['database']}`"))
            conn.commit()
        print(f"✅ Database '{db_config['database']}' ensured.")

        # Step 2: Connect to the target database
        db_url = f"{server_url}/{db_config['database']}"
        engine = create_engine(db_url)

        # Step 3: Create tables if they don't exist
        with engine.connect() as conn:
            # Students table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mva_10th_students (
                    Sr_No INT,
                    Roll_No VARCHAR(255),
                    Name VARCHAR(255),
                    English_Theory FLOAT,
                    English_Practical FLOAT,
                    English_Total FLOAT,
                    Hindi_Theory FLOAT,
                    Hindi_Practical FLOAT,
                    Hindi_Total FLOAT,
                    Maths_Theory FLOAT,
                    Maths_Practical FLOAT,
                    Maths_Total FLOAT,
                    Science_Theory FLOAT,
                    Science_Practical FLOAT,
                    Science_Total FLOAT,
                    Social_Science_Theory FLOAT,
                    Social_Science_Practical FLOAT,
                    Social_Science_Total FLOAT,
                    AI_Theory FLOAT,
                    AI_Practical FLOAT,
                    AI_Total FLOAT,
                    Grand_Total FLOAT,
                    Percentage FLOAT,
                    New_Old VARCHAR(10),
                    Cluster INT
                )
            """))

            # Summary table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mva_10th_summary (
                    metric VARCHAR(255),
                    value VARCHAR(255),
                    category VARCHAR(255)
                )
            """))

            # Full results table (with data_hash)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mva_10th_full_results (
                    run_id VARCHAR(255) PRIMARY KEY,
                    data_hash VARCHAR(255),
                    timestamp DATETIME,
                    results_json JSON
                )
            """))

            # Models metadata table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mva_10th_models (
                    model_name VARCHAR(255),
                    saved_at DATETIME,
                    run_id VARCHAR(255)
                )
            """))
            conn.commit()
        print("✅ All tables ensured.")

        # Step 4: Save data
        df_to_save = df.copy()
        if 'Cluster' in df_to_save.columns:
            df_to_save['Cluster'] = df_to_save['Cluster'].astype('Int64')
        df_to_save.to_sql("mva_10th_students", con=engine, if_exists='replace', index=False, method='multi')

        # Summary
        summary = []
        bs = results.get('basic_stats', {})
        summary.extend([
            {'metric': 'total_students', 'value': str(bs.get('total_students', 'N/A')), 'category': 'basic'},
            {'metric': 'avg_percentage', 'value': f"{bs.get('avg_percentage', 0):.2f}", 'category': 'basic'},
            {'metric': 'optimal_clusters', 'value': str(results.get('clustering', {}).get('optimal_k', 'N/A')), 'category': 'clustering'},
            {'metric': 'lr_r2', 'value': f"{results.get('linear_regression_r2', 0):.4f}", 'category': 'model'}
        ])
        pd.DataFrame(summary).to_sql("mva_10th_summary", con=engine, if_exists='replace', index=False)

        # Full results
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        full_record = pd.DataFrame([{
            'run_id': run_id,
            'data_hash': data_hash or 'unknown',
            'timestamp': datetime.now(),
            'results_json': json.dumps(results, default=str)
        }])
        full_record.to_sql("mva_10th_full_results", con=engine, if_exists='append', index=False)

        # Model metadata
        model_meta = [{'run_id': run_id, 'model_name': name} for name in models.keys()]
        if model_meta:
            pd.DataFrame(model_meta).to_sql("mva_10th_models", con=engine, if_exists='append', index=False)

        print("✅ All data saved to SQL.")
    except Exception as e:
        print(f"❌ SQL save error: {e}")
        raise

def fetch_past_runs_from_db(db_config):
    try:
        # First ensure DB exists (reuse logic)
        server_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}"
        server_engine = create_engine(server_url)
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_config['database']}`"))
            conn.commit()

        # Now connect to DB
        engine_url = f"{server_url}/{db_config['database']}"
        engine = create_engine(engine_url)
        query = "SELECT run_id, timestamp, data_hash FROM mva_10th_full_results ORDER BY timestamp DESC LIMIT 10"
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Failed to fetch past runs: {e}")
        return pd.DataFrame()

def check_data_exists_in_db(data_hash, db_config):
    try:
        engine_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(engine_url)
        query = text("SELECT run_id FROM mva_10th_full_results WHERE data_hash = :hash LIMIT 1")
        with engine.connect() as conn:
            result = conn.execute(query, {"hash": data_hash}).fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"DB check error: {e}")
        return None

def load_from_sql_by_run_id(run_id, db_config):
    try:
        engine_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(engine_url)
        df = pd.read_sql("SELECT * FROM mva_10th_students", engine)
        query = text("SELECT results_json FROM mva_10th_full_results WHERE run_id = :rid")
        with engine.connect() as conn:
            result = conn.execute(query, {"rid": run_id}).fetchone()
        if not result:
            return None, None, None
        results = json.loads(result[0])
        return df, results, {}  # Models not reloaded from DB (could be added via joblib if needed)
    except Exception as e:
        print(f"Load from SQL error: {e}")
        return None, None, None

def create_dashboard(df_parsed, analysis_results, trained_models=None):
    st.set_page_config(page_title="MVA Analysis Dashboard", layout="wide")
    st.title("🎓 MVA Results Dashboard - 8th to 10th Analysis")

    # Determine subject columns (exclude metadata)
    subject_cols = [col for col in df_parsed.columns if col.endswith('_Total')]

    page = st.sidebar.selectbox("View", ["Overview", "Subject Performance", "Correlation", "Outliers", "Clustering", "Model Insights"])

    if page == "Overview":
        st.write(f"**Total Students:** {len(df_parsed)}")
        st.write(f"**Analysis Date:** {analysis_results.get('analysis_date', 'N/A')}")
        st.dataframe(df_parsed.head(10))

    elif page == "Subject Performance":
        if subject_cols:
            # Show average scores as a bar chart (still useful for comparison)
            avg = df_parsed[subject_cols].mean().sort_values(ascending=False)
            st.subheader("Average Scores by Subject")
            st.plotly_chart(px.bar(avg, orientation='h', title="Avg Scores by Subject"))

            # Now allow user to pick a subject for detailed grade distribution (pie chart)
            subj = st.selectbox("Select Subject for Grade Distribution", subject_cols)

            # Define grade bands
            def get_grade(score):
                if pd.isna(score):
                    return 'Missing'
                elif score >= 91:
                    return 'A1 (91-100)'
                elif score >= 81:
                    return 'A2 (81-90)'
                elif score >= 71:
                    return 'B1 (71-80)'
                elif score >= 61:
                    return 'B2 (61-70)'
                elif score >= 51:
                    return 'C1 (51-60)'
                elif score >= 41:
                    return 'C2 (41-50)'
                elif score >= 33:
                    return 'D (33-40)'
                else:
                    return 'E (<33)'

            # Apply grading
            grade_series = df_parsed[subj].apply(get_grade)

            # Define custom order for grades (highest to lowest)
            grade_order = ['A1 (91-100)', 'A2 (81-90)', 'B1 (71-80)', 'B2 (61-70)', 'C1 (51-60)', 'C2 (41-50)', 'D (33-40)', 'E (<33)', 'Missing']

            # Convert to categorical with custom order
            grade_series = pd.Categorical(grade_series, categories=grade_order, ordered=True)

            # Count and sort by custom order
            grade_counts = grade_series.value_counts().reindex(grade_order).dropna()

            # Optional: Remove 'Missing' if you don't want it shown
            if 'Missing' in grade_counts.index:
                grade_counts = grade_counts.drop('Missing')

            if not grade_counts.empty:
                st.subheader(f"Grade Distribution: {subj}")

                # Define colors matching the NEW grade labels
                colors = {
                    'A1 (91-100)': '#FFD700',
                    'A2 (81-90)':  '#4CAF50',
                    'B1 (71-80)':  '#8BC34A',
                    'B2 (61-70)':  '#2196F3',
                    'C1 (51-60)':  '#FF9800',
                    'C2 (41-50)':  '#FF5722',
                    'D (33-40)':   '#9E9E9E',
                    'E (<33)':     '#F44336',
                }

                fig_pie = px.pie(
                    values=grade_counts.values,
                    names=grade_counts.index,
                    title=f"Grade Distribution in {subj}",
                    hole=0.3,
                    color=grade_counts.index,          # Required for color_discrete_map to work
                    color_discrete_map=colors
                )
                fig_pie.update_traces(sort=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No valid scores to display grade distribution.")

    elif page == "Correlation":
        if subject_cols:
            corr = df_parsed[subject_cols].corr()
            st.subheader("Correlation Matrix")
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto"), use_container_width=True)

            # --- Add interpretable insights ---
            st.subheader("Key Correlation Insights")

            # Unstack and filter strong correlations (|r| >= 0.5), exclude self-correlations
            corr_unstacked = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).unstack().dropna()
            strong_corr = corr_unstacked[abs(corr_unstacked) >= 0.5].sort_values(key=abs, ascending=False)

            if not strong_corr.empty:
                insight_data = []
                for (subj1, subj2), r in strong_corr.items():
                    strength = "Strong positive" if r > 0.7 else "Moderate positive" if r > 0.5 else "Moderate negative" if r < -0.5 else "Strong negative"
                    insight_data.append({
                        "Subject 1": subj1.replace('_Total', ''),
                        "Subject 2": subj2.replace('_Total', ''),
                        "Correlation (r)": round(r, 3),
                        "Interpretation": strength
                    })
                insight_df = pd.DataFrame(insight_data)
                st.dataframe(insight_df, use_container_width=True)
            else:
                st.info("No strong correlations (|r| ≥ 0.5) found between subjects.")

    elif page == "Outliers":
        for method in ['iqr', 'zscore', 'isolation_forest']:
            key = f'outliers_{method}'
            if key in analysis_results:
                st.subheader(f"{method.replace('_', ' ').title()} Outliers")
                st.dataframe(pd.DataFrame(analysis_results[key]))

    elif page == "Clustering":
        if 'clustering' in analysis_results:
            clust = analysis_results['clustering']
            st.write(f"**Optimal Clusters:** {clust['optimal_k']}")
            
            # --- 1. Cluster Summary as DataFrame ---
            cluster_summary = clust['cluster_summary']
            summary_df = pd.DataFrame(cluster_summary).T  # Transpose to have subjects as rows
            summary_df.index.name = 'Subject'
            summary_df = summary_df.reset_index()
            
            st.subheader("Cluster Performance Profiles")
            st.dataframe(summary_df)

            # --- 2. Visualize Cluster Profiles ---
            if not summary_df.empty:
                # Melt for plotting
                plot_df = summary_df.melt(id_vars='Subject', var_name='Cluster', value_name='Average Score')
                plot_df['Cluster'] = plot_df['Cluster'].astype(str)

                fig = px.bar(
                    plot_df,
                    x='Subject',
                    y='Average Score',
                    color='Cluster',
                    barmode='group',
                    title="Average Subject Scores by Cluster",
                    labels={'Average Score': 'Mean Score', 'Cluster': 'Student Group'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            # --- 3. Cluster Distribution (if df_parsed has 'Cluster' column) ---
            if 'Cluster' in df_parsed.columns:
                cluster_counts = df_parsed['Cluster'].value_counts().sort_index()
                dist_df = pd.DataFrame({
                    'Cluster': cluster_counts.index.astype(str),
                    'Number of Students': cluster_counts.values
                })
                st.subheader("Cluster Size Distribution")
                fig2 = px.pie(
                    dist_df,
                    values='Number of Students',
                    names='Cluster',
                    title="Student Distribution Across Clusters"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No clustering results available.")

    elif page == "Model Insights":
        if trained_models and 'linear_regression' in trained_models:
            lr = trained_models['linear_regression']
            feats = analysis_results.get('feature_names_for_modeling', [])
            if feats:
                coef_df = pd.DataFrame({'Subject': feats, 'Impact': lr.coef_}).sort_values('Impact', key=abs, ascending=False)
                st.plotly_chart(px.bar(coef_df, x='Impact', y='Subject', orientation='h'))
        else:
            st.info("No model insights available.")