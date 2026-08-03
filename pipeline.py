"""LeanDojo proof-style and mathematical-domain topic experiment.

The experiment deliberately separates two questions that the original clustering
prototype mixed together:

* style: which tactic moves and adjacent tactic pairs characterize a proof?
* domain: which explicitly referenced premises characterize its subject matter?

Both views use TF-IDF followed by non-negative matrix factorization (NMF).  NMF
gives every proof a mixture of topics instead of forcing a hard partition.  A
dominant topic is retained for coloring the dependency-free viewer.
"""

import json
import os
import re
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

RNG = 0
DATA = os.path.join("data", "leandojo_benchmark_4", "leandojo_benchmark_4")
OUT = "out"
TOPIC_CANDIDATES = [4, 6, 8, 10, 12, 14, 16]
STABILITY_REPEATS = 4
STABILITY_SAMPLE_FRAC = 0.80
MIN_STABILITY = 0.75
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------- loading
def load_theorems(paths):
    """Load every theorem with at least one traced tactic from ``paths``."""
    kept = []
    source_counts = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        selected = [t for t in data if t.get("traced_tactics")]
        kept.extend(selected)
        source_counts[os.path.basename(path)] = len(selected)
    return kept, source_counts


# --------------------------------------------------------------- tokenization
TACTIC_HEAD = re.compile(r"^[A-Za-z_][A-Za-z0-9_'!?]*")


def tactic_name(tactic):
    match = TACTIC_HEAD.match(tactic.strip())
    return match.group(0) if match else "<anon>"


def style_document(theorem):
    """Tactic-head unigrams plus adjacent tactic-head bigrams."""
    heads = [tactic_name(step["tactic"]) for step in theorem["traced_tactics"]]
    tokens = ["TAC_" + head for head in heads]
    tokens.extend("BIGRAM_" + a + "__" + b for a, b in zip(heads, heads[1:]))
    return " ".join(tokens)


def domain_document(theorem):
    """Explicit premise full names and their top-level namespaces.

    LeanDojo's annotations mark identifiers referenced in tactic syntax.  They do
    not enumerate every internal lemma used by automation such as ``simp``.
    """
    tokens = []
    for step in theorem["traced_tactics"]:
        annotated = step.get("annotated_tactic") or [None, []]
        for premise in annotated[1]:
            full_name = premise.get("full_name")
            if full_name:
                tokens.append("PREM_" + full_name)
                if "." in full_name:
                    tokens.append("NS_" + full_name.split(".")[0])
    return " ".join(tokens)


def tfidf_matrix(documents):
    vectorizer = TfidfVectorizer(
        token_pattern=r"\S+", min_df=2, sublinear_tf=True, lowercase=False
    )
    return vectorizer.fit_transform(documents), vectorizer


# ------------------------------------------------------------ topic modelling
def fit_nmf(X, n_topics, seed=RNG):
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        solver="cd",
        max_iter=600,
        tol=1e-4,
        random_state=seed,
    )
    weights = model.fit_transform(X)
    return model, weights


def aligned_topic_similarity(reference, candidate):
    """Mean cosine similarity after optimal one-to-one topic alignment."""
    similarities = cosine_similarity(reference, candidate)
    rows, cols = linear_sum_assignment(-similarities)
    return float(similarities[rows, cols].mean())


def topic_diagnostics(X, candidates=TOPIC_CANDIDATES):
    """Reconstruction and repeated-subsample component-stability curves.

    Each subsample fit uses 80% of the proofs without replacement while keeping
    the full vocabulary.  Topics are aligned to a full-data reference fit with
    the Hungarian algorithm, then compared by cosine similarity.
    """
    squared_norm = float(X.multiply(X).sum())
    rng = np.random.RandomState(RNG)
    diagnostics = []
    for k in candidates:
        reference, _ = fit_nmf(X, k, RNG)
        similarities = []
        sample_size = max(k + 1, int(X.shape[0] * STABILITY_SAMPLE_FRAC))
        for repeat in range(STABILITY_REPEATS):
            indices = rng.choice(X.shape[0], size=sample_size, replace=False)
            candidate, _ = fit_nmf(X[indices], k, RNG + repeat + 1)
            similarities.append(
                aligned_topic_similarity(reference.components_, candidate.components_)
            )
        diagnostics.append(
            {
                "k": k,
                "relative_reconstruction_error": float(
                    reference.reconstruction_err_ / np.sqrt(squared_norm)
                ),
                "stability": float(np.mean(similarities)),
                "stability_sd": float(np.std(similarities, ddof=1)),
            }
        )
        print(
            f"  topics={k:2d} reconstruction={diagnostics[-1]['relative_reconstruction_error']:.4f} "
            f"stability={diagnostics[-1]['stability']:.3f}"
        )
    return diagnostics


def reconstruction_elbow(diagnostics, min_stability=MIN_STABILITY):
    """Select the reconstruction-curve elbow among acceptably stable fits.

    This is a transparent resolution heuristic, not an estimator of a true
    number of latent proof kinds.
    """
    eligible = [d for d in diagnostics if d["stability"] >= min_stability]
    if len(eligible) < 3:
        return max(diagnostics, key=lambda d: d["stability"])["k"]
    x = np.array([d["k"] for d in eligible], dtype=float)
    y = np.array([d["relative_reconstruction_error"] for d in eligible], dtype=float)
    xn = (x - x.min()) / (np.ptp(x) + 1e-12)
    yn = (y - y.min()) / (np.ptp(y) + 1e-12)
    # Perpendicular distance to the endpoint chord; endpoints have distance 0.
    distance = np.abs(
        (yn[-1] - yn[0]) * xn
        - (xn[-1] - xn[0]) * yn
        + xn[-1] * yn[0]
        - yn[-1] * xn[0]
    )
    distance /= np.hypot(yn[-1] - yn[0], xn[-1] - xn[0]) + 1e-12
    return int(x[int(np.argmax(distance))])


def normalized_mixture(weights):
    totals = weights.sum(axis=1, keepdims=True)
    mixture = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    dominant = np.full(weights.shape[0], -1, dtype=int)
    signal = totals[:, 0] > 0
    dominant[signal] = np.argmax(mixture[signal], axis=1)
    entropy = np.zeros(weights.shape[0], dtype=float)
    if weights.shape[1] > 1:
        safe = np.where(mixture > 0, mixture, 1.0)
        entropy[signal] = -np.sum(mixture[signal] * np.log(safe[signal]), axis=1)
        entropy[signal] /= np.log(weights.shape[1])
    return mixture, dominant, entropy


def display_term(term):
    if term.startswith("TAC_"):
        return term[4:]
    if term.startswith("BIGRAM_"):
        return term[7:].replace("__", " → ")
    if term.startswith("PREM_"):
        return term[5:]
    if term.startswith("NS_"):
        return "namespace " + term[3:]
    return term


def describe_topics(model, vectorizer, mixture, dominant, theorems, view):
    terms = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic_id, component in enumerate(model.components_):
        order = np.argsort(-component)
        ranked = terms[order]
        top_terms = [display_term(term) for term in ranked[:10]]
        if view == "style":
            tactic_indices = [i for i in order if terms[i].startswith("TAC_")]
            tactics = [display_term(terms[i]) for i in tactic_indices[:6]]
            bigrams = [display_term(t) for t in ranked if t.startswith("BIGRAM_")][:5]
            leading_weight = component[tactic_indices[0]]
            label_terms = [
                display_term(terms[i])
                for i in tactic_indices[:3]
                if component[i] >= 0.25 * leading_weight
            ]
        else:
            tactics = []
            bigrams = []
            premises = [display_term(t) for t in ranked if t.startswith("PREM_")][:6]
            namespaces = [display_term(t) for t in ranked if t.startswith("NS_")][:4]
            label_terms = premises[:3] or namespaces[:3]
        members = dominant == topic_id
        representatives = np.argsort(-mixture[:, topic_id])[:5]
        topic = {
            "id": int(topic_id),
            "size": int(members.sum()),
            "label": " · ".join(label_terms) or f"topic {topic_id}",
            "top_terms": top_terms,
            "mean_len": float(
                np.mean(
                    [len(theorems[i]["traced_tactics"]) for i in np.where(members)[0]]
                )
            )
            if members.any()
            else 0.0,
            "representatives": [theorems[i]["full_name"] for i in representatives],
        }
        if view == "style":
            topic.update({"top_tactics": tactics, "top_bigrams": bigrams})
        else:
            topic.update({"top_premises": premises, "top_namespaces": namespaces})
        topics.append(topic)
    return topics


# -------------------------------------------------------------- visualization
def layouts(X):
    n_components = min(64, X.shape[0] - 1, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RNG)
    embedding = svd.fit_transform(X)
    embedding = Normalizer(copy=False).fit_transform(embedding)
    pca = PCA(n_components=3, random_state=RNG)
    pca_points = pca.fit_transform(embedding)
    tsne_points = TSNE(
        n_components=3,
        perplexity=30,
        init="pca",
        random_state=RNG,
        max_iter=1000,
        metric="cosine",
    ).fit_transform(embedding)

    def scale(points):
        points = points - points.mean(axis=0)
        return points / (np.abs(points).max() + 1e-12) * 50.0

    return scale(pca_points), scale(tsne_points), {
        "svd_components": n_components,
        "svd_explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
        "pca3_evr": [float(v) for v in pca.explained_variance_ratio_],
    }


# ---------------------------------------------------------------------- driver
def main():
    paths = [os.path.join(DATA, "random", name) for name in ("val.json", "test.json")]
    theorems, source_counts = load_theorems(paths)
    print(f"loaded {len(theorems)} tactic proofs: {source_counts}")

    style_docs = [style_document(t) for t in theorems]
    domain_docs = [domain_document(t) for t in theorems]
    X_style, style_vectorizer = tfidf_matrix(style_docs)
    X_domain, domain_vectorizer = tfidf_matrix(domain_docs)
    domain_signal = np.asarray(X_domain.getnnz(axis=1) > 0)
    print(f"style matrix {X_style.shape}, nnz={X_style.nnz}")
    print(
        f"domain matrix {X_domain.shape}, nnz={X_domain.nnz}, "
        f"proofs with signal={domain_signal.sum()}"
    )

    print("style topic diagnostics")
    style_diagnostics = topic_diagnostics(X_style)
    style_k = reconstruction_elbow(style_diagnostics)
    print("domain topic diagnostics")
    domain_diagnostics = topic_diagnostics(X_domain)
    domain_k = reconstruction_elbow(domain_diagnostics)
    print(f"selected exploratory resolutions: style={style_k}, domain={domain_k}")

    style_model, style_weights = fit_nmf(X_style, style_k)
    domain_model, domain_weights = fit_nmf(X_domain, domain_k)
    style_mix, style_dominant, style_entropy = normalized_mixture(style_weights)
    domain_mix, domain_dominant, domain_entropy = normalized_mixture(domain_weights)
    style_signal = style_dominant >= 0

    style_topics = describe_topics(
        style_model, style_vectorizer, style_mix, style_dominant, theorems, "style"
    )
    domain_topics = describe_topics(
        domain_model, domain_vectorizer, domain_mix, domain_dominant, theorems, "domain"
    )

    pca_points, tsne_points, layout_stats = layouts(X_style)
    module_labels = np.array(["/".join(t["file_path"].split("/")[:2]) for t in theorems])
    style_ami = float(
        adjusted_mutual_info_score(
            module_labels[style_signal], style_dominant[style_signal]
        )
    )
    domain_ami = float(
        adjusted_mutual_info_score(
            module_labels[domain_signal], domain_dominant[domain_signal]
        )
    )

    points = []
    for i, theorem in enumerate(theorems):
        style_order = np.argsort(-style_mix[i])[:3] if style_signal[i] else []
        domain_order = np.argsort(-domain_mix[i])[:3] if domain_signal[i] else []
        points.append(
            {
                "i": i,
                "name": theorem["full_name"],
                "file": theorem["file_path"],
                "n": len(theorem["traced_tactics"]),
                "tactics": [tactic_name(s["tactic"]) for s in theorem["traced_tactics"]][
                    :40
                ],
                "script": "\n".join(s["tactic"] for s in theorem["traced_tactics"])[
                    :1200
                ],
                "c": int(style_dominant[i]),
                "domain_c": int(domain_dominant[i]),
                "style_entropy": round(float(style_entropy[i]), 4),
                "domain_entropy": round(float(domain_entropy[i]), 4),
                "style_mix": [
                    [int(j), round(float(style_mix[i, j]), 4)] for j in style_order
                ],
                "domain_mix": [
                    [int(j), round(float(domain_mix[i, j]), 4)] for j in domain_order
                ],
                "pca": [round(float(v), 3) for v in pca_points[i]],
                "tsne": [round(float(v), 3) for v in tsne_points[i]],
            }
        )

    views = {
        "style": {
            "k": style_k,
            "topics": style_topics,
            "unclassified": int((~style_signal).sum()),
        },
        "domain": {
            "k": domain_k,
            "topics": domain_topics,
            "unclassified": int((~domain_signal).sum()),
        },
    }
    payload = {
        "points": points,
        "views": views,
        # Backward-compatible aliases used by older viewer builds.
        "clusters": style_topics,
        "k": style_k,
    }

    proof_lengths = np.array([len(t["traced_tactics"]) for t in theorems])
    stats = {
        "n_theorems": len(theorems),
        "source_counts": source_counts,
        "style": {
            "matrix_shape": list(X_style.shape),
            "nnz": int(X_style.nnz),
            "density": float(X_style.nnz / np.prod(X_style.shape)),
            "diagnostics": style_diagnostics,
            "selected_topics": style_k,
            "topics": style_topics,
            "n_with_topic": int(style_signal.sum()),
            "mean_mixture_entropy": float(style_entropy[style_signal].mean()),
            "module_ami": style_ami,
        },
        "domain": {
            "matrix_shape": list(X_domain.shape),
            "nnz": int(X_domain.nnz),
            "density": float(X_domain.nnz / np.prod(X_domain.shape)),
            "n_with_signal": int(domain_signal.sum()),
            "diagnostics": domain_diagnostics,
            "selected_topics": domain_k,
            "topics": domain_topics,
            "mean_mixture_entropy": float(domain_entropy[domain_signal].mean()),
            "module_ami": domain_ami,
        },
        "layout": layout_stats,
        "proof_len": {
            "mean": float(proof_lengths.mean()),
            "median": float(np.median(proof_lengths)),
            "min": int(proof_lengths.min()),
            "max": int(proof_lengths.max()),
        },
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "proofs.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    os.makedirs("app", exist_ok=True)
    with open(os.path.join("app", "data.js"), "w", encoding="utf-8") as f:
        f.write("window.PROOF_DATA = ")
        json.dump(payload, f)
        f.write(";\n")
    print("wrote out/proofs.json, out/stats.json, and app/data.js")


if __name__ == "__main__":
    main()
