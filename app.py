import io
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import h5py
import sqlite3
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve)
from lightgbm import LGBMClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
SEED = 42

st.set_page_config(page_title="실시간 불량 탐지 시스템", layout="wide")
st.title("실시간 불량 탐지 시스템")
st.markdown("---")

def load_file(file):
    name = file.name
    content = file.read()
    if name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(content))
    elif name.endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(content), engine='openpyxl')
    elif name.endswith('.xls'):
        try:
            return pd.read_excel(io.BytesIO(content), engine='xlrd')
        except:
            return pd.read_csv(io.BytesIO(content), sep='\t')
    elif name.endswith('.parquet'):
        return pd.read_parquet(io.BytesIO(content))
    elif name.endswith('.json'):
        return pd.read_json(io.BytesIO(content))
    elif name.endswith(('.h5', '.hdf5')):
        with h5py.File(io.BytesIO(content), 'r') as f:
            key = list(f.keys())[0]
            return pd.DataFrame(f[key][:])
    elif name.endswith(('.db', '.sqlite')):
        tmp_path = f'/tmp/{name}'
        with open(tmp_path, 'wb') as f:
            f.write(content)
        conn = sqlite3.connect(tmp_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        df = pd.read_sql(f"SELECT * FROM {tables['name'][0]}", conn)
        conn.close()
        return df
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None

ALLOWED = ["csv", "xlsx", "xls", "parquet", "feather", "json", "h5", "hdf5", "db", "sqlite"]

st.subheader("📂 데이터 업로드")
col_up1, col_up2 = st.columns(2)
with col_up1:
    train_file = st.file_uploader("학습 데이터", type=ALLOWED)
with col_up2:
    test_file = st.file_uploader("진단 데이터", type=ALLOWED)

if train_file and test_file:
    df_train = load_file(train_file)
    df_test  = load_file(test_file)
    if df_train is None or df_test is None:
        st.stop()

    if 'Time' in df_train.columns:
        df_train = df_train.drop(columns=['Time'])
    if 'Time' in df_test.columns:
        df_test = df_test.drop(columns=['Time'])

    y_train_all = df_train['Pass/Fail'].map({-1: 0, 1: 1}).values
    feat_train  = df_train.drop(columns=['Pass/Fail'])
    has_label   = 'Pass/Fail' in df_test.columns
    y_test      = df_test['Pass/Fail'].map({-1: 0, 1: 1}).values if has_label else None
    feat_test   = df_test.drop(columns=['Pass/Fail'], errors='ignore')

    sensor_names = feat_train.columns.tolist()
    X_train = feat_train.values.astype(np.float32)
    X_test  = feat_test.values.astype(np.float32)

    # ════════════════════════════════
# 1차 필터링
# ════════════════════════════════
st.markdown("---")
st.subheader("🔵 1차 필터링")

with st.spinner("1차 필터링 실행 중..."):
    # 3대 핵심 센서 임계값
    T_65  = 34.79
    T_205 = 13.43
    T_510 = 60.84

    # 숫자 강제 변환
    feat_test_1st = feat_test.copy()
    for col in ['65', '205', '510']:
        if col in feat_test_1st.columns:
            feat_test_1st[col] = pd.to_numeric(feat_test_1st[col], errors='coerce')

    # AND 조건 판별
    has_cols = all(c in feat_test_1st.columns for c in ['65', '205', '510'])
    if not has_cols:
        st.error("진단 데이터에 '65', '205', '510' 컬럼이 없습니다.")
        st.stop()

    condition = (
        (feat_test_1st['65']  >= T_65)  &
        (feat_test_1st['205'] >= T_205) &
        (feat_test_1st['510'] >= T_510)
    )
    pred1 = condition.values

    # 불량 원인 텍스트 생성
    def make_reason(row):
        return (
            f"🚨 [3대 핵심센서 동시초과] "
            f"챔버온도(65번): {row['65']:.1f} (기준:{T_65}) | "
            f"가스압력(205번): {row['205']:.1f} (기준:{T_205}) | "
            f"배기펌프(510번): {row['510']:.1f} (기준:{T_510})"
        )

    reasons = []
    for i, row in feat_test_1st.iterrows():
        if condition.loc[i]:
            reasons.append(make_reason(row))
        else:
            reasons.append("이상 없음")

# 결과 표시
n_suspect1 = pred1.sum()
st.metric("🚨 1차 불량 의심 샘플", f"{n_suspect1}개")

st.dataframe(pd.DataFrame({
    "샘플 번호":   range(len(pred1)),
    "1차 판정":    ["🚨 1차 불량 (확인요망)" if p else "✅ 정상 통과" for p in pred1],
    "불량 상세원인": reasons
}), use_container_width=True)

# 2차 필터링에서 prob1 변수가 필요하므로 대체값 생성
prob1 = pred1.astype(float)

    # ════════════════════════════════
    # 2차 필터링
    # ════════════════════════════════
    st.markdown("---")
    st.subheader("🟠 2차 필터링")

    with st.spinner("2차 필터링 실행 중..."):
        imputer2 = SimpleImputer(strategy='median')
        X_train2 = imputer2.fit_transform(X_train)
        X_test2  = imputer2.transform(X_test)

        mu = X_train2.mean(axis=0)
        sg = X_train2.std(axis=0) + 1e-8

        def make_sensor_features(X, mu, sg):
            feats = {}
            X_z = (X - mu) / sg
            for i in range(X.shape[1]):
                feats[f'raw_{i}'] = X[:, i]
                feats[f'z_{i}']   = X_z[:, i]
            feats['row_mean']   = X.mean(axis=1)
            feats['row_std']    = X.std(axis=1)
            feats['row_min']    = X.min(axis=1)
            feats['row_max']    = X.max(axis=1)
            feats['row_q25']    = np.quantile(X, 0.25, axis=1)
            feats['row_q75']    = np.quantile(X, 0.75, axis=1)
            feats['row_iqr']    = feats['row_q75'] - feats['row_q25']
            feats['row_energy'] = (X**2).mean(axis=1)
            seg = max(1, X.shape[1]//4)
            for k in range(4):
                block = X[:, k*seg:(k+1)*seg]
                if block.shape[1] > 0:
                    feats[f'seg{k}_mean'] = block.mean(axis=1)
                    feats[f'seg{k}_std']  = block.std(axis=1)
            diff = np.diff(X_z, axis=1)
            if diff.shape[1] > 0:
                feats['diff_mean']     = diff.mean(axis=1)
                feats['diff_std']      = diff.std(axis=1)
                feats['diff_abs_mean'] = np.abs(diff).mean(axis=1)
                feats['diff_abs_max']  = np.abs(diff).max(axis=1)
            return pd.DataFrame(feats)

        def make_cross_features(X, sensor_names, top_cols):
            feats = {}
            col_idx = {name: i for i, name in enumerate(sensor_names)}
            pairs_done = set()
            for i, ca in enumerate(top_cols):
                for cb in top_cols[i+1:]:
                    if (ca, cb) in pairs_done: continue
                    pairs_done.add((ca, cb))
                    ia, ib = col_idx[ca], col_idx[cb]
                    va, vb = X2[:, ia], X2[:, ib]
                    feats[f'diff_{ca}_{cb}']  = va - vb
                    feats[f'absd_{ca}_{cb}']  = np.abs(va - vb)
                    feats[f'ratio_{ca}_{cb}'] = va / (np.abs(vb) + 1e-8)
            return pd.DataFrame(feats) if feats else pd.DataFrame(index=range(len(X)))

        X_sf_tr = make_sensor_features(X_train2, mu, sg)
        X_sf_te = make_sensor_features(X_test2,  mu, sg)
        top_cols = sensor_names[:min(5, len(sensor_names))]

        X2 = X_train2
        X_cf_tr = make_cross_features(X_train2, sensor_names, top_cols)
        X2 = X_test2
        X_cf_te = make_cross_features(X_test2,  sensor_names, top_cols)

        scaler2  = StandardScaler()
        X_sc2_tr = scaler2.fit_transform(X_train2)
        X_sc2_te = scaler2.transform(X_test2)
        pca2     = PCA(n_components=0.95, random_state=SEED)
        X_pca_tr = pca2.fit_transform(X_sc2_tr)
        X_pca_te = pca2.transform(X_sc2_te)
        ev       = pca2.explained_variance_
        t2_tr    = np.sum((X_pca_tr**2)/ev, axis=1)
        t2_te    = np.sum((X_pca_te**2)/ev, axis=1)
        spe_tr   = np.sum((X_sc2_tr - pca2.inverse_transform(X_pca_tr))**2, axis=1)
        spe_te   = np.sum((X_sc2_te - pca2.inverse_transform(X_pca_te))**2, axis=1)
        fdc_tr   = pd.DataFrame({'t2_score': t2_tr, 'spe_score': spe_tr,
                                  'log_t2': np.log1p(t2_tr), 'log_spe': np.log1p(spe_tr)})
        fdc_te   = pd.DataFrame({'t2_score': t2_te, 'spe_score': spe_te,
                                  'log_t2': np.log1p(t2_te), 'log_spe': np.log1p(spe_te)})

        X_tr_comb = np.hstack([X_sf_tr.values, X_cf_tr.values, fdc_tr.values]).astype(np.float32)
        X_te_comb = np.hstack([X_sf_te.values, X_cf_te.values, fdc_te.values]).astype(np.float32)
        all_names = X_sf_tr.columns.tolist() + X_cf_tr.columns.tolist() + fdc_tr.columns.tolist()

        df_temp  = pd.DataFrame(X_tr_comb, columns=all_names)
        corr_mat = df_temp.corr().abs()
        upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        to_drop  = [c for c in upper.columns if any(upper[c] > 0.95)]
        X_tr_filt = df_temp.drop(columns=to_drop)
        filt_names = X_tr_filt.columns.tolist()

        sel = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced')
        sel.fit(X_tr_filt.values, y_train_all)
        thresh_imp  = np.mean(sel.feature_importances_) * 0.7
        indices     = np.where(sel.feature_importances_ >= thresh_imp)[0]
        final_names = [filt_names[i] for i in indices]
        X_tr_fin    = X_tr_filt.values[:, indices]
        te_indices  = [all_names.index(n) for n in final_names]
        X_te_fin    = X_te_comb[:, te_indices]

        pos_weight  = (y_train_all==0).sum() / (y_train_all==1).sum()
        lgbm_params = dict(learning_rate=0.02, n_estimators=3000, num_leaves=31,
                           reg_lambda=3.0, min_child_samples=2, colsample_bytree=0.8,
                           subsample=0.8, random_state=SEED,
                           scale_pos_weight=pos_weight, verbose=-1)
        rf_params   = dict(n_estimators=1200, max_depth=8, min_samples_split=2,
                           min_samples_leaf=1, max_features='sqrt', n_jobs=-1,
                           random_state=SEED, class_weight={0:1, 1:pos_weight*1.95})

        tscv = TimeSeriesSplit(n_splits=5)
        test_lgbm_folds, test_rf_folds, fold_weights = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr_fin)):
            Xf_tr, Xf_val = X_tr_fin[tr_idx], X_tr_fin[val_idx]
            yf_tr, yf_val = y_train_all[tr_idx], y_train_all[val_idx]
            fold_weights.append(max(1, yf_tr.sum()))

            m_lgbm = LGBMClassifier(**lgbm_params)
            m_lgbm.fit(Xf_tr, yf_tr,
                       eval_set=[(Xf_val, yf_val)],
                       callbacks=[lgb.early_stopping(150, verbose=False),
                                  lgb.log_evaluation(-1)])
            test_lgbm_folds.append(m_lgbm.predict_proba(X_te_fin)[:,1])

            m_rf = RandomForestClassifier(**rf_params)
            m_rf.fit(Xf_tr, yf_tr)
            test_rf_folds.append(m_rf.predict_proba(X_te_fin)[:,1])

        fw = np.array(fold_weights, dtype=float); fw /= fw.sum()
        def rank_avg_w(preds, w):
            ranks = [rankdata(p)/len(p) for p in preds]
            return np.average(ranks, axis=0, weights=w)

        pred_lgbm = rank_avg_w(test_lgbm_folds, fw)
        pred_rf2  = rank_avg_w(test_rf_folds,   fw)
        prob2     = np.mean([rankdata(pred_lgbm)/len(pred_lgbm),
                             rankdata(pred_rf2)/len(pred_rf2)], axis=0)

    if has_label:
        prec_c, rec_c, thr_c = precision_recall_curve(y_test, prob2)
        valid    = rec_c[:-1] >= 0.90
        best_thr = thr_c[valid][np.argmax(prec_c[:-1][valid])] if valid.any() else thr_c[np.argmax(rec_c[:-1])]
        f2_scores = [(4*p*r)/(4*p+r+1e-8) for p,r in zip(prec_c[:-1], rec_c[:-1])]
        f2_thr   = thr_c[np.argmax(f2_scores)]
        pred2    = (prob2 >= min(best_thr, f2_thr)).astype(int)
    else:
        pred2 = (prob2 >= 0.5).astype(int)

    # 불량 탐지 수만 표시
    n_suspect2 = pred2.sum()
    st.metric("🚨 2차 불량 의심 샘플", f"{n_suspect2}개")

    st.dataframe(pd.DataFrame({
        "샘플 번호": range(len(prob2)),
        "2차 위험 점수": prob2.round(4),
        "2차 판정": ["🚨 불량 의심" if p else "✅ 정상" for p in pred2]
    }), use_container_width=True)

    # ════════════════════════════════
    # 근본원인 분석
    # ════════════════════════════════
    st.markdown("---")
    st.subheader("🔴 근본원인 분석")

    rf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                      random_state=SEED, n_jobs=-1)
    rf_final.fit(feat_train.fillna(feat_train.median()), y_train_all)
    importances   = rf_final.feature_importances_
    X_test_filled = feat_test.fillna(feat_train.median())
    probs_final   = rf_final.predict_proba(X_test_filled)[:,1]

    # 불량/요주의 샘플만 필터링해서 표시
    defect_results = []
    for idx in range(len(X_test_filled)):
        score = probs_final[idx]
        if score < 0.06:
            continue  # 정상은 표시 안 함
        elif score < 0.35:
            status = "⚠️ 요주의"
        else:
            status = "🚨 불량"
        contrib  = X_test_filled.iloc[idx].values * importances
        top3_idx = np.argsort(contrib)[-3:][::-1]
        top3     = [feat_test.columns[i] for i in top3_idx]
        defect_results.append({
            "샘플 번호": idx,
            "위험 점수": round(float(score), 3),
            "판정": status,
            "1순위 센서": top3[0],
            "2순위 센서": top3[1],
            "3순위 센서": top3[2],
        })

    if defect_results:
        df_defect = pd.DataFrame(defect_results)
        c1, c2 = st.columns(2)
        c1.metric("🚨 불량 샘플", len(df_defect[df_defect['판정']=='🚨 불량']))
        c2.metric("⚠️ 요주의 샘플", len(df_defect[df_defect['판정']=='⚠️ 요주의']))
        st.dataframe(df_defect, use_container_width=True)
    else:
        st.success("불량 또는 요주의 샘플이 없습니다!")

    # ════════════════════════════════
    # 종합 리포트
    # ════════════════════════════════
    if has_label:
        st.markdown("---")
        st.subheader("📋 종합 평가 리포트")
        sorted_idx   = np.argsort(probs_final)[::-1]
        total_test   = len(y_test)
        actual_fails = y_test.sum()
        top30 = int(total_test * 0.30)
        top50 = int(total_test * 0.50)

        pred_50 = np.zeros(total_test); pred_50[sorted_idx[:top50]] = 1
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, pred_50).ravel()
        rec_a = tp_a / actual_fails * 100
        pre_a = tp_a / (tp_a+fp_a) * 100 if (tp_a+fp_a)>0 else 0
        acc_a = (tp_a+tn_a) / total_test * 100

        pred_30 = np.zeros(total_test); pred_30[sorted_idx[:top30]] = 1
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, pred_30).ravel()
        rec_b = tp_b / actual_fails * 100
        pre_b = tp_b / (tp_b+fp_b) * 100 if (tp_b+fp_b)>0 else 0
        acc_b = (tp_b+tn_b) / total_test * 100

        st.markdown("#### ▶ 1. 운영 전략별 성능 비교")
        st.dataframe(pd.DataFrame({
            "운영 전략": ["최대 탐지 모드 (안전 우선)", "효율 우선 모드"],
            "불량 탐지율 (Recall)": [f"{rec_a:.1f}% ({actual_fails}개 중 {tp_a}개)",
                                     f"{rec_b:.1f}% ({actual_fails}개 중 {tp_b}개)"],
            "가짜 알람 (FP)": [f"{fp_a}건", f"{fp_b}건"],
            "알람 적중률 (Precision)": [f"{pre_a:.1f}%", f"{pre_b:.1f}%"],
            "전체 정확도 (Accuracy)": [f"{acc_a:.1f}%", f"{acc_b:.1f}%"],
            "현장 적용 가이드": ["불량을 절대 놓치면 안 되는 핵심 공정 적용",
                                 "인력이 부족하여 가짜 알람을 줄여야 할 때 적용"]
        }), use_container_width=True, hide_index=True)

        st.markdown("#### ▶ 2. 현장 실무 적용 효과")
        st.dataframe(pd.DataFrame({
            "검사 기준": ["상위 30%만 타겟 검사", "상위 50%만 타겟 검사"],
            "검사할 웨이퍼 물량": [f"{top30}장", f"{top50}장"],
            "색출한 실제 불량 수": [f"{tp_b}개 (전체의 {rec_b:.1f}%)",
                                    f"{tp_a}개 (전체의 {rec_a:.1f}%)"],
            "효율성 의미": [f"전체 물량의 30%만 검사해도 불량의 {rec_b:.0f}% 이상 차단",
                            f"전체 물량의 50%만 검사하면 불량의 {rec_a:.0f}% 이상 완벽 방어"]
        }), use_container_width=True, hide_index=True)

        st.info(f"""
**[Executive Summary]**

본 모델과 진단기를 현장에 도입하여 위험도 상위 30~50% 물량만 우선 검사(Target Inspection)할 경우,
기존 전수 검사 대비 현장의 검사 인력과 리소스를 절반 이하로 획기적으로 줄이면서도
치명적인 수율 저하 불량은 {rec_b:.0f}~{rec_a:.0f}% 이상 완벽하게 사전 방어할 수 있습니다.
        """)
    else:
        st.warning("진단 데이터에 'Pass/Fail' 컬럼이 없어 종합 리포트를 생성할 수 없습니다.")