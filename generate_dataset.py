
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
fs = 400
Ts_ms = 1000.0 / fs
DUR_MS_MIN, DUR_MS_MAX = 30, 1500
REST_MS_MIN, REST_MS_MAX = 40, 600
N_EPISODES = 3000
WIN_MS = 2000
WIN_N  = int(round(WIN_MS / Ts_ms))

cols = [
    "ratio_occupation",
    "nb_transitions",
    "duree_max_bloc_1_ms",
    "duree_activation_estimee_ms",
    "temps_repos_avant_ms",
    "temps_repos_apres_ms",
]

def simulate_episode():
    dur_ms = np.random.uniform(DUR_MS_MIN, DUR_MS_MAX)
    rest_before_ms = np.random.uniform(REST_MS_MIN, REST_MS_MAX)
    rest_after_ms  = np.random.uniform(REST_MS_MIN, REST_MS_MAX)
    dur_n   = max(1, int(round(dur_ms / Ts_ms)))
    rest_b  = max(1, int(round(rest_before_ms / Ts_ms)))
    rest_a  = max(1, int(round(rest_after_ms  / Ts_ms)))
    serie = [0]*rest_b + [1]*dur_n + [0]*rest_a
    if np.random.rand() < 0.15 and len(serie)>0:
        k = np.random.randint(0, len(serie))
        serie[k] = 1-serie[k]
    return {"serie": serie}

def window_episode(s):
    L = len(s)
    if L >= WIN_N:
        start = max(0, (L - WIN_N)//2)
        return s[start:start+WIN_N]
    else:
        pad = WIN_N - L
        left = pad // 2
        right = pad - left
        return ([0]*left) + s + ([0]*right)

def extract_features_from_serie(serie):
    x = np.array(serie, dtype=int)
    L = len(x)
    ratio_occupation = x.mean()
    transitions = int(np.sum(np.abs(np.diff(x))))
    if x.sum() > 0:
        idx = np.where(x == 1)[0]
        blocks = []
        start = idx[0]
        for i in range(1, len(idx)):
            if idx[i] != idx[i-1] + 1:
                blocks.append((start, idx[i-1]))
                start = idx[i]
        blocks.append((start, idx[-1]))
        max_block = max((b[1]-b[0]+1) for b in blocks)
        duree_max_bloc_1_ms = max_block * Ts_ms
        duree_activation_estimee_ms = duree_max_bloc_1_ms
        first1, last1 = blocks[0][0], blocks[-1][1]
        zeros_before = first1
        zeros_after  = L - (last1+1)
        temps_repos_avant_ms = zeros_before * Ts_ms
        temps_repos_apres_ms = zeros_after * Ts_ms
    else:
        duree_max_bloc_1_ms = 0.0
        duree_activation_estimee_ms = 0.0
        temps_repos_avant_ms = L * Ts_ms
        temps_repos_apres_ms = L * Ts_ms
    return [
        ratio_occupation,
        float(transitions),
        float(duree_max_bloc_1_ms),
        float(duree_activation_estimee_ms),
        float(temps_repos_avant_ms),
        float(temps_repos_apres_ms)
    ]

def compute_score(features):
    f = np.array(features, float)
    term = np.sin(f/7).sum() + np.cos(f/11).sum() + (np.abs(f)**0.3).sum()
    if len(f) >= 2:
        term += (f[0]*f[-1])/100
    if len(f) >= 3:
        term += (f[1]*f[2])/200
    return 0.4*term + np.random.normal(0, 0.5)

def make_labels_from_scores(scores, n_classes=3):
    scores = np.array(scores, float)
    if n_classes == 2:
        t = np.percentile(scores, 50)
        return (scores >= t).astype(int)
    t1, t2 = np.percentile(scores, [33, 66])
    labels = np.zeros_like(scores, int)
    labels[scores >= t1] = 1
    labels[scores >= t2] = 2
    return labels

def main():
    episodes = [simulate_episode() for _ in range(N_EPISODES)]
    windows = [window_episode(ep['serie']) for ep in episodes]
    feats = [extract_features_from_serie(w) for w in windows]
    scores = [compute_score(f) for f in feats]
    labels = make_labels_from_scores(scores, 3)
    df = pd.DataFrame(feats, columns=cols)
    df['score'] = scores
    df['label'] = labels
    # label -> nom de classe via médianes de durée
    med = df.groupby('label')['duree_activation_estimee_ms'].median().sort_values()
    order = list(med.index)
    mapping = {order[0]:'tap', order[1]:'swipe_rapide', order[2]:'swipe_lent'}
    df['class_name'] = df['label'].map(mapping)
    # Split & scaler
    X = df[cols].values
    y = df['label'].values
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    Xmin = X_train.min(axis=0); Xmax = X_train.max(axis=0)
    scaler = {'feature_names': cols, 'Xmin': Xmin.tolist(), 'Xmax': Xmax.tolist()}
    df.to_csv('microgestes_dataset.csv', index=False)
    with open('scaler_params.json','w', encoding='utf-8') as f:
        json.dump(scaler, f, indent=2, ensure_ascii=False)
    print(' microgestes_dataset.csv et scaler_params.json générés')

if __name__ == '__main__':
    main()
