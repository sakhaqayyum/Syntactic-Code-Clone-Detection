"""
Syntactic code clone detection on BigCloneBench: 1D-CNN vs. conventional ML.

Pipeline
  1. Preprocessing : comment removal -> lexical analysis/tokenization -> parsing (AST)
                     -> code fragment (method) extraction
  2. Threshold-based similarity analysis : flags duplicate / near-duplicate fragments
  3. 1D-CNN (siamese) learning discriminative features from token sequences
  4. Comparison with conventional ML (LogReg, NaiveBayes, KNN, DecisionTree,
     RandomForest, SVM) on hand-crafted similarity features
  5. Metrics: accuracy, precision, recall, F1 (+ results.csv)

Dataset (CodeXGLUE BigCloneBench format):
  dataset/data.jsonl  -> {"func": "<java code>", "idx": "123"}
  dataset/train.txt | valid.txt | test.txt -> "idx1<TAB>idx2<TAB>label"
  https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench

Install : pip install torch scikit-learn pandas numpy javalang
Run     : python bcb_clone_pipeline.py --data_dir dataset
"""
import argparse, difflib, json, random, re, time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import javalang
except ImportError:
    javalang = None
    print("[warn] javalang not installed -> AST features disabled (pip install javalang)")

PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||\+\+|--|[^\s\w]")
KEYWORDS = set("""abstract assert boolean break byte case catch char class const continue default
do double else enum extends final finally float for goto if implements import instanceof int
interface long native new package private protected public return short static strictfp super
switch synchronized this throw throws transient try void volatile while true false null""".split())


# ----------------------------------------------------------------------------
# 1. PREPROCESSING: lexical analysis, tokenization, parsing, fragment extraction
# ----------------------------------------------------------------------------
def strip_comments(code):
    return re.sub(r"//.*?$|/\*.*?\*/", " ", code, flags=re.S | re.M)


def lex(code):
    """Lexical analysis + tokenization. Identifiers -> <ID>, numbers -> <NUM>
    so that renamed (Type-2) clones map to identical token streams."""
    out = []
    for t in TOKEN_RE.findall(strip_comments(code)):
        if t in KEYWORDS or not (t[0].isalnum() or t[0] == "_"):
            out.append(t)
        elif t[0].isdigit():
            out.append("<NUM>")
        else:
            out.append("<ID>")
    return out


def parse_and_extract(code):
    """Parsing + code-fragment extraction.
    Wraps the snippet in a dummy class, parses it with javalang and returns
    (AST node-type sequence, number of extracted method fragments)."""
    if javalang is None:
        return [], 1
    try:
        tree = javalang.parse.parse("class W {\n" + strip_comments(code) + "\n}")
        methods = [n for _, n in tree.filter(javalang.tree.MethodDeclaration)]
        ast_seq = [type(n).__name__ for _, n in tree]
        return ast_seq, max(len(methods), 1)
    except Exception:
        return [], 1


def preprocess(path):
    recs = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            code = d["func"]
            toks = lex(code)
            ast_seq, n_frag = parse_and_extract(code)
            recs[str(d["idx"])] = dict(
                toks=toks,
                tset=set(toks),
                grams=set(zip(toks, toks[1:], toks[2:])),
                ast=Counter(ast_seq),
                lines=max(code.count("\n"), 1),
                n_frag=n_frag,
                text=" ".join(toks),
            )
    return recs


def load_pairs(path, recs, n, seed):
    pairs = []
    with open(path) as f:
        for line in f:
            a, b, y = line.strip().split("\t")
            if a in recs and b in recs:
                pairs.append((a, b, int(y)))
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


# ----------------------------------------------------------------------------
# 2. SIMILARITY FEATURES + THRESHOLD-BASED DUPLICATE / NEAR-DUPLICATE ANALYSIS
# ----------------------------------------------------------------------------
def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def multiset_jaccard(a, b):
    keys = set(a) | set(b)
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return num / den if den else 1.0


def pair_features(pairs, recs, tfidf_rows):
    X = []
    for a, b, _ in pairs:
        ra, rb = recs[a], recs[b]
        seq = difflib.SequenceMatcher(None, ra["toks"][:150], rb["toks"][:150],
                                      autojunk=False).ratio()
        cos = float(tfidf_rows[a].multiply(tfidf_rows[b]).sum())  # rows are L2-normalised
        la, lb = len(ra["toks"]), len(rb["toks"])
        X.append([
            jaccard(ra["tset"], rb["tset"]),
            jaccard(ra["grams"], rb["grams"]),
            cos,
            seq,
            multiset_jaccard(ra["ast"], rb["ast"]),
            min(la, lb) / max(la, lb, 1),
            min(ra["lines"], rb["lines"]) / max(ra["lines"], rb["lines"]),
            abs(ra["n_frag"] - rb["n_frag"]),
        ])
    return np.array(X, dtype=np.float32)


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return dict(accuracy=accuracy_score(y, p), precision=pr, recall=rc, f1=f1)


def threshold_analysis(Xtr, ytr, Xte, yte, near_thr=0.70, dup_thr=0.95):
    """Threshold on the combined similarity score (mean of token 3-gram Jaccard,
    TF-IDF cosine and sequence ratio). Pairs >= dup_thr are duplicates,
    near_thr <= s < dup_thr near-duplicates; the best clone threshold is tuned on train."""
    score = lambda X: X[:, [1, 2, 3]].mean(axis=1)
    s_tr, s_te = score(Xtr), score(Xte)
    best_t = max(np.linspace(0.05, 0.99, 95),
                 key=lambda t: metrics(ytr, (s_tr >= t).astype(int))["f1"])
    pred = (s_te >= best_t).astype(int)
    print(f"\n[Threshold analysis] best clone threshold = {best_t:.2f}")
    print(f"  duplicates (s >= {dup_thr})              : {(s_te >= dup_thr).sum()}")
    print(f"  near-duplicates ({near_thr} <= s < {dup_thr})   : "
          f"{((s_te >= near_thr) & (s_te < dup_thr)).sum()}")
    print(f"  non-duplicate (s < {near_thr})           : {(s_te < near_thr).sum()}")
    return metrics(yte, pred)


# ----------------------------------------------------------------------------
# 3. 1D-CNN (siamese) on tokenised source code
# ----------------------------------------------------------------------------
def build_vocab(recs, min_freq=2, max_size=20000):
    cnt = Counter(t for r in recs.values() for t in r["toks"])
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for t, c in cnt.most_common(max_size):
        if c >= min_freq:
            vocab[t] = len(vocab)
    return vocab


def encode(toks, vocab, max_len):
    ids = [vocab.get(t, UNK) for t in toks[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


class CNN1D(nn.Module):
    def __init__(self, vocab_size, emb=128, filters=128, kernels=(3, 5, 7), drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels)
        self.drop = nn.Dropout(drop)
        d = filters * len(kernels)
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(128, 1))

    def encode(self, x):
        h = self.emb(x).transpose(1, 2)
        feats = [F.relu(c(h)).max(dim=2).values for c in self.convs]
        return self.drop(torch.cat(feats, dim=1))

    def forward(self, a, b):
        va, vb = self.encode(a), self.encode(b)
        return self.head(torch.cat([(va - vb).abs(), va * vb], dim=1)).squeeze(1)


def to_tensors(pairs, recs, vocab, max_len):
    A = torch.tensor([encode(recs[a]["toks"], vocab, max_len) for a, _, _ in pairs])
    B = torch.tensor([encode(recs[b]["toks"], vocab, max_len) for _, b, _ in pairs])
    y = torch.tensor([p[2] for p in pairs], dtype=torch.float)
    return TensorDataset(A, B, y)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for a, b, _ in loader:
        out += (torch.sigmoid(model(a.to(device), b.to(device))) > 0.5).long().cpu().tolist()
    return np.array(out)


def train_cnn(tr, va, te, vocab_size, args, device):
    model = CNN1D(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tl = DataLoader(tr, args.batch_size, shuffle=True)
    vl, tel = DataLoader(va, 256), DataLoader(te, 256)
    best, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0
        for a, b, y in tl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(a, b), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        f1 = metrics(va.tensors[2].numpy(), predict(model, vl, device))["f1"]
        print(f"  epoch {ep}/{args.epochs}  loss={tot/len(tl):.4f}  val-F1={f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return metrics(te.tensors[2].numpy(), predict(model, tel, device))


# ----------------------------------------------------------------------------
# 4. MAIN: compare 1D-CNN with conventional ML
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_valid", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preprocessing (lexing, tokenizing, parsing, fragment extraction)...")
    recs = preprocess(f"{args.data_dir}/data.jsonl")
    print(f"  {len(recs)} code fragments; parsed ASTs for "
          f"{sum(1 for r in recs.values() if r['ast'])}")

    tr = load_pairs(f"{args.data_dir}/train.txt", recs, args.n_train, args.seed)
    va = load_pairs(f"{args.data_dir}/valid.txt", recs, args.n_valid, args.seed)
    te = load_pairs(f"{args.data_dir}/test.txt", recs, args.n_test, args.seed)
    ytr, yte = np.array([p[2] for p in tr]), np.array([p[2] for p in te])
    print(f"  pairs: train={len(tr)} valid={len(va)} test={len(te)} "
          f"(clone ratio train={ytr.mean():.2f})")

    # similarity features
    ids = list(recs)
    tfidf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2),
                            max_features=50000, sublinear_tf=True)
    M = tfidf.fit_transform([recs[i]["text"] for i in ids]).tocsr()
    rows = {i: M[k] for k, i in enumerate(ids)}
    print("Computing pairwise similarity features...")
    Xtr, Xte = pair_features(tr, recs, rows), pair_features(te, recs, rows)

    results = {}
    results["Threshold-based similarity"] = threshold_analysis(Xtr, ytr, Xte, yte)

    # conventional ML
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Naive Bayes": GaussianNB(),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=args.seed),
        "Random Forest": RandomForestClassifier(300, n_jobs=-1, random_state=args.seed),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=2.0)),
    }
    for name, m in models.items():
        print(f"Training {name}...")
        m.fit(Xtr, ytr)
        results[name] = metrics(yte, m.predict(Xte))

    # 1D-CNN
    print("Training 1D-CNN...")
    vocab = build_vocab(recs)
    t0 = time.time()
    results["1D-CNN (proposed)"] = train_cnn(
        to_tensors(tr, recs, vocab, args.max_len),
        to_tensors(va, recs, vocab, args.max_len),
        to_tensors(te, recs, vocab, args.max_len),
        len(vocab), args, device)
    print(f"  CNN time: {time.time() - t0:.0f}s on {device}")

    df = pd.DataFrame(results).T.round(4).sort_values("f1", ascending=False)
    print("\n================ TEST RESULTS ================")
    print(df.to_string())
    df.to_csv("results.csv")
    print("\nSaved results.csv")


if __name__ == "__main__":
    main()
