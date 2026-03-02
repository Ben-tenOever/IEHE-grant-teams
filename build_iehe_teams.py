import json, re, math
from itertools import combinations

ALLOWED = {"DARPA","NIH","NSF","AHRQ","CDC"}

GENERIC_STOP = set("""
a an the and or of to in for with on by from into over under across within via using use used
this that these those is are was were be been being as at it its their they them we our
have has had will would should can could may might
research health clinical community care medicine medical based work professor investigator funded nyu
program programs project projects center centers department departments
""".split())

DOMAIN_ANCHORS = [
    ("hypertension_cardiovascular", ["hypertension","cardiovascular","heart","bp","blood pressure","cvd"]),
    ("mental_health", ["mental","depression","anxiety","psychi","substance","addiction"]),
    ("sleep", ["sleep","insomnia","apnea","circadian"]),
    ("equity", ["equity","disparit","underserv","rac","latinx","black","minority","social determinant","sdoH".lower()]),
    ("implementation", ["implementation","dissemination","scale","adoption","fidelity","pragmatic","real world","hybrid"]),
]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(text: str):
    text = norm(text)
    toks = re.findall(r"[a-z]{3,}", text)
    return [t for t in toks if t not in GENERIC_STOP]

def tfidf_vectors(texts):
    docs = [tokenize(t) for t in texts]
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1.0 for w in df}
    vecs = []
    for d in docs:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        v = {w: (tf[w] / len(d)) * idf.get(w, 0.0) for w in tf} if d else {}
        vecs.append(v)
    return vecs, idf

def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv is not None:
            dot += av * bv
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def top_overlap_terms(a, b, k=6):
    shared = []
    for w in set(a.keys()) & set(b.keys()):
        shared.append((a[w] + b[w], w))
    shared.sort(reverse=True)
    return [w for _, w in shared[:k]]

def pick_domain(text):
    t = norm(text)
    for label, keys in DOMAIN_ANCHORS:
        for k in keys:
            if k in t:
                return label
    return "implementation"

def aims_for(domain):
    if domain == "hypertension_cardiovascular":
        return (
            "Aim 1: Identify implementation barriers and inequities affecting hypertension control across care settings and populations | "
            "Aim 2: Deploy a pragmatic implementation strategy to improve blood pressure control and cardiovascular risk management | "
            "Aim 3: Evaluate reach, fidelity, sustainment, and cost impact using equity stratified outcomes"
        )
    if domain == "mental_health":
        return (
            "Aim 1: Quantify unmet need and disparities in mental health access and outcomes using mixed methods and stakeholder input | "
            "Aim 2: Implement and test a scalable care pathway intervention using a pragmatic or hybrid effectiveness implementation design | "
            "Aim 3: Measure adoption, engagement, clinical outcomes, and sustainment with equity focused metrics"
        )
    if domain == "sleep":
        return (
            "Aim 1: Define gaps and inequities in sleep assessment and treatment across target populations and delivery settings | "
            "Aim 2: Implement a pragmatic intervention to improve sleep identification and care linkage within routine workflows | "
            "Aim 3: Evaluate effectiveness, adherence, and sustainability including downstream health outcomes"
        )
    if domain == "equity":
        return (
            "Aim 1: Characterize structural and workflow drivers of inequities in care delivery and outcomes within the target context | "
            "Aim 2: Co design and implement an equity centered intervention using community and health system partnerships | "
            "Aim 3: Test impact on reach, effectiveness, adoption, and sustainment using standardized implementation outcomes"
        )
    return (
        "Aim 1: Define the implementation context, barriers, facilitators, and inequities using mixed methods | "
        "Aim 2: Adapt and implement an evidence based intervention using pragmatic or hybrid study designs | "
        "Aim 3: Evaluate effectiveness, reach, adoption, fidelity, and sustainment with equity stratified outcomes"
    )

def main():
    faculty = json.load(open("data/faculty_index.json"))
    opps = json.load(open("data/opportunities.json"))

    faculty_text = []
    faculty_name = []
    for f in faculty:
        faculty_name.append(f.get("name","").strip())
        blob = " ".join([
            f.get("name",""),
            f.get("title",""),
            f.get("summary",""),
            " ".join(f.get("keywords",[]) if isinstance(f.get("keywords",[]), list) else [])
        ])
        faculty_text.append(blob)

    faculty_vecs, _ = tfidf_vectors(faculty_text)

    out = []
    kept_opps = 0

    for o in opps:
        agency = (o.get("agency_name") or o.get("agency") or "").strip()
        if agency not in ALLOWED:
            continue

        kept_opps += 1
        title = o.get("opportunity_title") or o.get("title") or ""
        desc = o.get("synopsis") or o.get("description") or ""
        opp_blob = f"{title} {desc}"
        opp_vec, _ = tfidf_vectors([opp_blob])
        ov = opp_vec[0]

        scores = []
        for i, fv in enumerate(faculty_vecs):
            s = cosine(fv, ov)
            scores.append((s, i))
        scores.sort(reverse=True)

        topN = [i for s,i in scores[:8] if s > 0.0] or [scores[0][1], scores[1][1]]

        # build pair candidates
        pair_cands = []
        for i,j in combinations(topN, 2):
            s_i = cosine(faculty_vecs[i], ov)
            s_j = cosine(faculty_vecs[j], ov)
            cohesion = cosine(faculty_vecs[i], faculty_vecs[j])
            team_score = 0.6 * ((s_i + s_j)/2.0) + 0.4 * cohesion
            pair_cands.append((team_score, (i,j)))
        pair_cands.sort(reverse=True)

        # build 3 person candidates
        tri_cands = []
        for i,j,k in combinations(topN, 3):
            s = (cosine(faculty_vecs[i], ov) + cosine(faculty_vecs[j], ov) + cosine(faculty_vecs[k], ov)) / 3.0
            coh = (cosine(faculty_vecs[i], faculty_vecs[j]) + cosine(faculty_vecs[i], faculty_vecs[k]) + cosine(faculty_vecs[j], faculty_vecs[k])) / 3.0
            team_score = 0.6 * s + 0.4 * coh
            tri_cands.append((team_score, (i,j,k)))
        tri_cands.sort(reverse=True)

        domain = pick_domain(title + " " + desc)
        aims = aims_for(domain)

        def emit(member_idx_tuple, size_cat):
            mem_names = [faculty_name[i] for i in member_idx_tuple]
            overlaps = []
            for i in member_idx_tuple:
                overlaps.extend(top_overlap_terms(faculty_vecs[i], ov, k=3))
            overlaps = [w for w in overlaps if w not in GENERIC_STOP]
            overlaps = list(dict.fromkeys(overlaps))[:8]

            out.append({
                "agency_name": agency,
                "opportunity_id": o.get("opportunity_id") or o.get("id"),
                "opportunity_number": o.get("opportunity_number") or o.get("number"),
                "opportunity_title": title,
                "opportunity_url": o.get("opportunity_url") or o.get("url"),
                "open_date": o.get("open_date"),
                "close_date": o.get("close_date"),
                "opp_status": o.get("opp_status") or o.get("status"),
                "team_members": "; ".join(mem_names),
                "team_size_category": size_cat,
                "proposed_title": f"{title}: implementation and equity focused evaluation",
                "rationale": (
                    f"Selected to align with the opportunity by combining complementary implementation and outcomes expertise across "
                    f"{', '.join(overlaps) if overlaps else 'IEHE domain anchors'}. The team composition supports pragmatic evaluation, "
                    f"equity stratified outcomes, and scalable dissemination."
                ),
                "aims_or_projects": aims,
            })

        for _, ij in pair_cands[:5]:
            emit(ij, "small")

        for _, ijk in tri_cands[:5]:
            emit(ijk, "large")

    json.dump(out, open("data/teams.json","w"), indent=2)
    print("kept_opps:", kept_opps)
    print("wrote teams:", len(out))
    print("pairs:", sum(1 for t in out if t["team_size_category"]=="small"))
    print("teams_3plus:", sum(1 for t in out if t["team_size_category"]=="large"))

if __name__ == "__main__":
    main()
