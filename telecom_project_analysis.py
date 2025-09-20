"""

Full telecom analysis 


"""

import os, argparse, json
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
import joblib
try:
    from sqlalchemy import create_engine
except Exception:
    create_engine = None

from logger import get_logger
logger = get_logger()

# Column detection heuristics
COLUMN_CANDIDATES = {
    'msisdn': ['msisdn','subscriber','user_id','id','imsi','msisdn_number'],
    'application': ['application','app','service','app_name'],
    'dl_bytes': ['dl_bytes','download_bytes','downlink_bytes','dl','bytes_dl','bytes_down'],
    'ul_bytes': ['ul_bytes','upload_bytes','uplink_bytes','ul','bytes_ul','bytes_up'],
    'session_duration': ['duration','session_duration','time','session_time','duration_seconds'],
    'handset': ['handset','handset_type','device','device_type','phone_model','model'],
    'manufacturer': ['manufacturer','vendor','brand','handset_manufacturer','make'],
    'tcp_retrans': ['tcp_retransmission','tcp_retrans','retransmission','tcp_retrans_count'],
    'rtt': ['rtt','round_trip_time','round_trip','round_trip_latency'],
    'throughput': ['throughput','avg_throughput','throughput_kbps','throughput_bps','tcp_throughput']
}

def find_column(df_columns, candidates):
    cols_lower = {c.lower(): c for c in df_columns}
    for cand in candidates:
        for col_lower, original in cols_lower.items():
            if cand.lower() in col_lower:
                return original
    return None

def auto_map_columns(df):
    mapped = {}
    for key, candidates in COLUMN_CANDIDATES.items():
        mapped[key] = find_column(df.columns, candidates)
    return mapped

# -------------------------
# Task 1: Data load & preprocessing 
# -------------------------
def load_data(path='telecomdata.csv'):
    logger.info(f"Loading data from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded dataframe with shape: {df.shape}")
    return df

def basic_cleaning(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            numeric_like = sample.str.replace('[, ]','', regex=True).str.match(r'^-?\d+(\.\d+)?$').all() if len(sample)>0 else False
            if numeric_like:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    return df

def treat_outliers_iqr(df, numeric_cols=None):
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            mean_val = df[col].mean()
            mask = (df[col] < lower) | (df[col] > upper)
            if mask.any():
                df.loc[mask, col] = mean_val
        except Exception:
            continue
    return df

# -------------------------
# Task 2: User Overview & EDA
# -------------------------
def task1_user_overview(df, mapping, output_dir='outputs', show_plots=True):
    os.makedirs(output_dir, exist_ok=True)
    msisdn = mapping.get('msisdn')
    handset_col = mapping.get('handset')
    manufacturer_col = mapping.get('manufacturer')
    app_col = mapping.get('application')
    dl_col = mapping.get('dl_bytes')
    ul_col = mapping.get('ul_bytes')
    duration_col = mapping.get('session_duration')

    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if handset_col and handset_col in df.columns:
        top10_handsets = df[handset_col].value_counts().head(10)
        results['top10_handsets'] = top10_handsets
        if manufacturer_col and manufacturer_col in df.columns:
            top3_manufacturers = df[manufacturer_col].value_counts().head(3)
            results['top3_manufacturers'] = top3_manufacturers
            top5_by_manufacturer = {}
            for m in top3_manufacturers.index:
                sub = df[df[manufacturer_col]==m]
                top5_by_manufacturer[m] = sub[handset_col].value_counts().head(5)
            results['top5_by_manufacturer'] = top5_by_manufacturer

    if msisdn and app_col and dl_col and ul_col and duration_col and all(c in df.columns for c in [msisdn, app_col, dl_col, ul_col, duration_col]):
        df = df.copy()
        df['total_bytes'] = df[dl_col] + df[ul_col]
        agg = df.groupby([msisdn, app_col]).agg(
            sessions_count=(app_col,'count'),
            total_duration=(duration_col,'sum'),
            total_dl=(dl_col,'sum'),
            total_ul=(ul_col,'sum'),
            total_bytes=('total_bytes','sum')
        ).reset_index()
        results['user_app_agg'] = agg
        user_overall = df.groupby(msisdn).agg(
            sessions_count=(app_col,'count'),
            total_duration=(duration_col,'sum'),
            total_dl=(dl_col,'sum'),
            total_ul=(ul_col,'sum'),
            total_bytes=('total_bytes','sum')
        ).reset_index()
        results['user_overall'] = user_overall
        user_overall.to_csv(os.path.join(output_dir,'aggregated_user_metrics.csv'), index=False)
    else:
        logger.warning('Missing columns for per-user application aggregation; skipping user_app_agg')

    results['dtypes'] = df.dtypes.apply(lambda x: str(x)).to_dict()
    basic_metrics = compute_basic_metrics(df, numeric_cols)
    results['basic_numeric_metrics'] = basic_metrics

    if show_plots:
        for c in numeric_cols:
            try:
                plt.figure(figsize=(6,4)); plt.hist(df[c].dropna(), bins=30); plt.title(f'Histogram of {c}'); plt.xlabel(c); plt.ylabel('Count'); plt.tight_layout(); plt.show()
                plt.figure(figsize=(6,3)); plt.boxplot(df[c].dropna()); plt.title(f'Boxplot of {c}'); plt.tight_layout(); plt.show()
            except Exception as e:
                logger.warning(f\"Plot failure for {c}: {e}\")

    if 'total_bytes' in df.columns:
        try:
            app_vars = [c for c in df.columns if c not in numeric_cols and c!=msisdn][:10]
            for app_col_bi in app_vars:
                if pd.api.types.is_numeric_dtype(df[app_col_bi]):
                    plt.figure(figsize=(6,4)); plt.scatter(df[app_col_bi], df['total_bytes'], alpha=0.4); plt.title(f'{app_col_bi} vs total_bytes'); plt.xlabel(app_col_bi); plt.ylabel('total_bytes'); plt.tight_layout(); plt.show()
        except Exception:
            pass

    if 'user_overall' in results and 'total_duration' in results['user_overall'].columns:
        try:
            uo = results['user_overall'].copy()
            uo['decile'] = pd.qcut(uo['total_duration'], 10, labels=False, duplicates='drop')
            decile_agg = uo.groupby('decile')['total_bytes'].sum().reset_index()
            results['decile_agg'] = decile_agg
            if show_plots:
                plt.figure(figsize=(6,4)); plt.bar(decile_agg['decile'].astype(str), decile_agg['total_bytes']); plt.title('Total bytes per duration decile'); plt.xlabel('decile'); plt.ylabel('total_bytes'); plt.tight_layout(); plt.show()
        except Exception as e:
            logger.warning(f\"Decile computation failed: {e}\")

    apps = [c for c in df.columns if any(k in c.lower() for k in ['social','google','email','youtube','netflix','gaming','other'])]
    if len(apps) >= 2:
        try:
            corr = df[apps].corr()
            results['app_correlation'] = corr
            if show_plots:
                plt.figure(figsize=(6,5)); plt.imshow(corr, interpolation='none', aspect='auto'); plt.colorbar(); plt.xticks(range(len(apps)), apps, rotation=90); plt.yticks(range(len(apps)), apps); plt.title('Correlation matrix (apps)'); plt.tight_layout(); plt.show()
            n_comp = min(3, len(apps))
            pca = PCA(n_components=n_comp); X = df[apps].fillna(0).values; scaler = StandardScaler(); Xs = scaler.fit_transform(X); comps = pca.fit_transform(Xs)
            results['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
            if show_plots and n_comp >= 2:
                plt.figure(figsize=(6,5)); plt.scatter(comps[:,0], comps[:,1], alpha=0.4); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA PC1 vs PC2'); plt.tight_layout(); plt.show()
        except Exception as e:
            logger.warning(f\"Correlation/PCA failed: {e}\")

    return results

# -------------------------
# Support functions
# -------------------------
def compute_basic_metrics(df, numeric_cols):
    meta = {}
    for col in numeric_cols:
        s = df[col]
        try:
            meta[col] = {'count': int(s.count()), 'mean': float(s.mean()), 'median': float(s.median()), 'std': float(s.std()), 'var': float(s.var()), 'min': float(s.min()), 'max': float(s.max()), 'skew': float(s.skew()), 'kurtosis': float(s.kurtosis())}
        except Exception:
            meta[col] = {}
    return meta

def pairplot_numeric(df, sample_size=500, show_plots=True):
    try:
        from pandas.plotting import scatter_matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            sample_df = df[numeric_cols].sample(min(sample_size, len(df)), random_state=42)
            scatter_matrix(sample_df, figsize=(12,12), alpha=0.3)
            plt.suptitle('Pairwise Scatter Matrix of Numeric Features')
            if show_plots:
                plt.show()
    except Exception as e:
        logger.warning(f\"Pairplot generation failed: {e}\")

# -------------------------
# Task 3: Engagement Analysis
# -------------------------
def task2_engagement(user_overall_df, output_dir='outputs', show_plots=True):
    df = user_overall_df.copy()
    metrics = [c for c in ['sessions_count','total_duration','total_bytes'] if c in df.columns]
    if len(metrics) < 1:
        logger.warning('Not enough engagement metrics found.')
        return df, None, None, None, None

    X = df[metrics].fillna(0).values
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Ks = list(range(1,8)); inertias = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42); km.fit(Xs); inertias.append(km.inertia_)
    if show_plots:
        plt.figure(figsize=(6,4)); plt.plot(Ks, inertias, marker='o'); plt.title('Elbow - engagement'); plt.xlabel('k'); plt.ylabel('inertia'); plt.tight_layout(); plt.show()
    kmeans = KMeans(n_clusters=3, random_state=42); df['engagement_cluster'] = kmeans.fit_predict(Xs)
    cluster_stats = df.groupby('engagement_cluster')[metrics].agg(['min','max','mean','sum']).reset_index()
    if show_plots:
        for m in metrics:
            try:
                groups = [df[df['engagement_cluster']==c][m].dropna() for c in sorted(df['engagement_cluster'].unique())]
                plt.figure(figsize=(6,4)); plt.boxplot(groups, labels=[str(c) for c in sorted(df['engagement_cluster'].unique())]); plt.title(f'{m} by engagement cluster'); plt.xlabel('cluster'); plt.ylabel(m); plt.tight_layout(); plt.show()
            except Exception:
                continue
    top10 = {m: df.nlargest(10, m)[[m]].reset_index() for m in metrics if m in df.columns}
    return df, kmeans, scaler, cluster_stats, top10

# -------------------------
# Task 4: Experience Analysis
# -------------------------
def task3_experience(df_raw, mapping, output_dir='outputs', show_plots=True):
    msisdn = mapping.get('msisdn')
    tcp_col = mapping.get('tcp_retrans'); rtt_col = mapping.get('rtt'); throughput_col = mapping.get('throughput'); handset_col = mapping.get('handset')
    if not msisdn:
        logger.warning('MSISDN not detected; skipping experience aggregation'); return None, None, None, None
    # build user_exp with available columns
    cols = {}
    if tcp_col in df_raw.columns: cols['avg_tcp'] = (tcp_col,'mean')
    if rtt_col in df_raw.columns: cols['avg_rtt'] = (rtt_col,'mean')
    if throughput_col in df_raw.columns: cols['avg_throughput'] = (throughput_col,'mean')
    if len(cols)==0:
        logger.warning('No experience metrics present'); return None, None, None, None
    agg_dict = {}
    for outcol, (src,func) in cols.items():
        agg_dict[outcol] = (src, 'mean')
    user_exp = df_raw.groupby(msisdn).agg(**{k:(v[0],'mean') for k,v in cols.items()}).reset_index()
    if handset_col in df_raw.columns:
        handsets = df_raw.groupby(msisdn)[handset_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
        user_exp = user_exp.merge(handsets, on=msisdn, how='left')
    for c in ['avg_tcp','avg_rtt','avg_throughput']:
        if c in user_exp.columns:
            user_exp[c] = user_exp[c].fillna(user_exp[c].mean())
    top_bottom = {}
    for c in ['avg_tcp','avg_rtt','avg_throughput']:
        if c in user_exp.columns:
            top_bottom[c] = {'top10': user_exp.nlargest(10,c)[[msisdn,c]], 'bottom10': user_exp.nsmallest(10,c)[[msisdn,c]]}
    exp_metrics = [c for c in ['avg_tcp','avg_rtt','avg_throughput'] if c in user_exp.columns]
    if len(exp_metrics)>0:
        X = user_exp[exp_metrics].fillna(0).values; scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42); user_exp['experience_cluster'] = kmeans.fit_predict(Xs)
        if show_plots and len(exp_metrics)>=2:
            try:
                plt.figure(figsize=(6,5)); plt.scatter(Xs[:,0], Xs[:,1], c=user_exp['experience_cluster'], alpha=0.4); plt.title('Experience clusters'); plt.tight_layout(); plt.show()
            except Exception:
                pass
    else:
        kmeans=None; scaler=None
    return user_exp, kmeans, scaler, top_bottom

# -------------------------
# Task 4: Satisfaction ANalysis
# -------------------------
def task4_satisfaction(user_overall_df, engagement_kmeans, engagement_scaler, user_exp_df, exp_kmeans, exp_scaler, mapping, output_dir='outputs', show_plots=True, mysql_url=None, save_model=False):
    msisdn = mapping.get('msisdn')
    if msisdn is None:
        logger.warning('No msisdn mapping; cannot compute satisfaction'); return None, None, None, None, None
    eng_metrics = [c for c in ['sessions_count','total_duration','total_bytes'] if c in user_overall_df.columns]
    if len(eng_metrics)==0:
        logger.warning('No engagement metrics'); return None, None, None, None, None
    X_eng = engagement_scaler.transform(user_overall_df[eng_metrics].fillna(0).values)
    eng_centroids = engagement_kmeans.cluster_centers_
    centroid_sums = eng_centroids.sum(axis=1); least_engaged_cluster = int(np.argmin(centroid_sums))
    least_centroid = eng_centroids[least_engaged_cluster].reshape(1,-1)
    eng_dists = pairwise_distances(X_eng, least_centroid).reshape(-1)
    user_overall_df['engagement_score'] = eng_dists
    exp_metrics = [c for c in ['avg_tcp','avg_rtt','avg_throughput'] if c in user_exp_df.columns]
    X_exp = exp_scaler.transform(user_exp_df[exp_metrics].fillna(0).values)
    exp_centroids = exp_kmeans.cluster_centers_
    exp_scores = exp_centroids[:,0] + exp_centroids[:,1] - exp_centroids[:,2]
    worst_exp_cluster = int(np.argmax(exp_scores)); worst_centroid = exp_centroids[worst_exp_cluster].reshape(1,-1)
    exp_dists = pairwise_distances(X_exp, worst_centroid).reshape(-1)
    user_exp_df['experience_score'] = exp_dists
    merged = user_overall_df[[msisdn,'engagement_score']].merge(user_exp_df[[msisdn,'experience_score']], on=msisdn, how='inner')
    merged['satisfaction_score'] = merged[['engagement_score','experience_score']].mean(axis=1)
    merged = merged.sort_values('satisfaction_score', ascending=False)
    combined = user_overall_df.merge(user_exp_df[[msisdn]+exp_metrics], on=msisdn, how='inner')
    if 'satisfaction_score' in merged.columns:
        combined = combined.merge(merged[[msisdn,'satisfaction_score']], on=msisdn, how='inner')
    feat_cols = eng_metrics + exp_metrics
    combined = combined.fillna(0)
    model=None; metrics=None
    if len(combined) > 5 and len(feat_cols)>0:
        X = combined[feat_cols].values; y = combined['satisfaction_score'].values
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42); model.fit(X_train, y_train)
        y_pred = model.predict(X_test); rmse = mean_squared_error(y_test, y_pred, squared=False); r2 = r2_score(y_test, y_pred)
        metrics = {'rmse': rmse, 'r2': r2}
        if save_model:
            joblib.dump(model, os.path.join(output_dir, f'rf_satisfaction_{datetime.utcnow().strftime(\"%Y%m%d%H%M%S\")}.joblib'))
    X_2 = merged[['engagement_score','experience_score']].values
    if len(X_2) >= 2:
        k2 = KMeans(n_clusters=2, random_state=42).fit(X_2); merged['satisfaction_cluster'] = k2.labels_
        if show_plots:
            plt.figure(figsize=(6,5)); plt.scatter(merged['engagement_score'], merged['experience_score'], c=merged['satisfaction_cluster'], alpha=0.5); plt.xlabel('engagement_score'); plt.ylabel('experience_score'); plt.title('Satisfaction clusters (k=2)'); plt.tight_layout(); plt.show()
    else:
        merged['satisfaction_cluster'] = np.nan
    try:
        cluster_avgs = merged.groupby('satisfaction_cluster')[['satisfaction_score','engagement_score','experience_score']].mean().reset_index()
    except Exception:
        cluster_avgs = None
    if mysql_url and create_engine:
        try:
            engine = create_engine(mysql_url); merged.to_sql('telecom_user_satisfaction', engine, if_exists='replace', index=False); logger.info('Exported to MySQL')
        except Exception as e:
            logger.warning(f'MySQL export failed: {e}')
    return merged, model, metrics, merged.head(10), cluster_avgs

# -------------------------
# Runner
# -------------------------
def run_full_pipeline(data_path='telecomdata.csv', output_dir='outputs', mysql_url=None, show_plots=True, save_model=False):
    os.makedirs(output_dir, exist_ok=True)
    df_raw = load_data(data_path)
    df_raw = basic_cleaning(df_raw)
    df_raw = treat_outliers_iqr(df_raw)
    mapping = auto_map_columns(df_raw)
    logger.info('Auto-detected mapping:'); logger.info(json.dumps(mapping, indent=2))
    t1 = task1_user_overview(df_raw, mapping, output_dir=output_dir, show_plots=show_plots)
    user_overall = t1.get('user_overall') if t1 else None
    if user_overall is None:
        logger.warning('user_overall missing; attempting fallback grouping by msisdn')
        msisdn = mapping.get('msisdn')
        if msisdn and msisdn in df_raw.columns:
            user_overall = df_raw.groupby(msisdn).size().reset_index(name='sessions_count')
        else:
            user_overall = pd.DataFrame()
    if not user_overall.empty:
        eng_df, eng_kmeans, eng_scaler, eng_cluster_stats, eng_top10 = task2_engagement(user_overall, output_dir=output_dir, show_plots=show_plots)
    else:
        eng_df=None; eng_kmeans=None; eng_scaler=None
    user_exp_df, exp_kmeans, exp_scaler, exp_top_bottom = task3_experience(df_raw, mapping, output_dir=output_dir, show_plots=show_plots)
    if eng_kmeans is not None and exp_kmeans is not None and user_exp_df is not None and not user_overall.empty:
        merged, model, metrics, top10, cluster_avgs = task4_satisfaction(user_overall, eng_kmeans, eng_scaler, user_exp_df, exp_kmeans, exp_scaler, mapping, output_dir=output_dir, show_plots=show_plots, mysql_url=mysql_url, save_model=save_model)
        try:
            merged.to_csv(os.path.join(output_dir, 'satisfaction_scores.csv'), index=False)
            top10.to_csv(os.path.join(output_dir, 'top10_satisfaction.csv'), index=False)
            if cluster_avgs is not None:
                cluster_avgs.to_csv(os.path.join(output_dir, 'satisfaction_cluster_avgs.csv'), index=False)
        except Exception:
            pass
    else:
        logger.warning('Skipping Task 4 due to missing components')
    logger.info('Pipeline completed. Plots were shown via plt.show().')
    return {'t1': t1, 'user_overall': user_overall, 'eng': {'df': locals().get('eng_df')}, 'exp': {'df': locals().get('user_exp_df')}, 'satisfaction': {'merged': locals().get('merged'), 'metrics': locals().get('metrics')}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telecom pipeline (reads telecomdata.csv).')
    parser.add_argument('--data_path', default='telecomdata.csv', help='CSV path (default telecomdata.csv)')
    parser.add_argument('--output_dir', default='outputs', help='Output dir')
    parser.add_argument('--mysql_url', default=None, help='Optional MySQL SQLAlchemy URL')
    parser.add_argument('--no_plots', action='store_true', help='Do not show plots')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    args = parser.parse_args()
    run_full_pipeline(data_path=args.data_path, output_dir=args.output_dir, mysql_url=args.mysql_url, show_plots=not args.no_plots, save_model=args.save_model)
