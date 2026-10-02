"""
master_basic.py
---------------
Generic first-pass coder. Reads data/sentences.csv (from fetch_first5.py, or any CSV with the
same columns) and produces, per document, a table like Srinidhi's coding example:

    output/statements.csv    one row per candidate statement:
                             statement | attribute | deontic (strength) | aim | object | conditions |
                             or else | incentive | statement type | lifecycle | flags
    output/edges.csv         issuer -> actor edges weighted by deontic strength (incl. UNSPECIFIED)
    output/diagnostics.csv   per-document anomaly checks + a verdict:
                             "generic OK" or what kind of custom handling the document needs
    output/first_pass.xlsx   all three tables in one workbook

It is deliberately generic: no site-specific rules. The diagnostics tell you where that breaks.

Run:  python master_basic.py                      (reads data/sentences.csv)
      python master_basic.py --input other.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import spacy

OUT = Path("output")

# ---------------- codebook (Srinidhi's scale + decision rules) ----------------
MARKERS = [  # (regex, strength, type) - first match wins at a position
    (r"\bmust not\b|\bshall not\b|\b(?:is|are) prohibited\b", 4, "prohibition"),
    (r"\bmust\b|\bshall\b|\b(?:is|are) required to\b|\bneeds? to\b|\bha(?:s|ve) to\b|\bmandatory\b", 4, "obligation"),
    (r"\bshould(?: not)?\b|\b(?:is|are) expected to\b|\bought to\b", 3, "recommendation"),
    (r"\bencourag\w*|\brecommend\w*|\binvit\w*|\burg(?:e|es|ed|ing)\b|\b(?:is|are) advised to\b|\bconsider\b", 2, "encouragement"),
    (r"\bmay\b|\bcan\b|\bcould\b|\bmight\b", 2, "permission/possibility"),
    (r"\b(?:allows?|enables?|lets?|helps?)\b(?=[^.]{0,60}\bto\b)", 2, "enabling"),
]
SELF_COMMIT = re.compile(r"\b(?:we|the ncsc|ncsc|dsit|(?:the )?(?:uk )?government)(?:\s+and\s+\w+)?\s+will\b"
                         r"(?!\s+(?:almost|likely|probably))", re.I)
ACTORS = {
    "developers": r"\bdevelopers?\b|\bpractitioners?\b|\bengineers?\b|\bdata scientists?\b",
    "providers": r"\bproviders?\b",
    "operators / deployers": r"\boperators?\b|\bdeployers?\b",
    "data custodians": r"\bdata custodians?\b|\bdata owners?\b",
    "maintainers / OSS community": r"\bmaintainers?\b|\bopen[- ]source (?:community|projects?|foundations?)\b",
    "platforms / repositories": r"\bhubs?\b|\brepositor(?:y|ies)\b|\bregistr(?:y|ies)\b|\bplatforms?\b",
    "vendors / suppliers": r"\bvendors?\b|\bsuppliers?\b|\bmanufacturers?\b",
    "integrators": r"\bintegrators?\b",
    "purchasers": r"\bpurchas\w*|\bbuyers?\b|\bprocur\w+|\bcustomers?\b",
    "end users": r"\bend[- ]users?\b|\busers?\b",
    "organisations": r"\borgani[sz]ations?\b|\bbusinesses\b|\bcompanies\b|\benterprises\b|\bsmes?\b",
    "supply chain / stakeholders": r"\bsupply chain\b|\bstakeholders?\b",
    "government": r"\bgovernments?\b|\bdepartments?\b|\bncsc\b|\bdsit\b",
    "regulators": r"\bregulators?\b|\bauthorit(?:y|ies)\b",
    "industry / academia / partners": r"\bindustry\b|\bacademia\b|\bpartners\b",
    "attackers (not an addressee)": r"\battackers?\b|\badversar\w+|\bthreat actors?\b|\bcriminals?\b",
    "you (document audience)": r"\byou\b|\byour\b",
}
ACTOR_RE = {k: re.compile(v, re.I) for k, v in ACTORS.items()}
ANY_ACTOR = re.compile("|".join(ACTORS.values()), re.I)
LIFECYCLE = {
    "design": r"\bdesign\w*|threat model",
    "training data": r"training data|\bdatasets?\b|\bprovenance\b|\bpoison\w*|data collection",
    "training / weights / fine-tuning": r"\bfine[- ]tun\w*|\bweights?\b|\bpre-?train\w*|\btrain(?:ing|ed)? (?:the |a |your )?models?\b",
    "development": r"\bdevelop\w*|\bsource code\b|\bdependenc\w+|\bsbom\b|\bsupply chain\b|\blibrar(?:y|ies)\b",
    "evaluation / testing": r"\bevaluat\w+|\btest\w*|\bred[- ]team",
    "deployment": r"\bdeploy\w*|\bin production\b|\bintegrat\w+",
    "operation / maintenance": r"\boperat(?:e|ed|ing|ion)\b|\bmonitor\w*|\bpatch\w*|\bupdat\w+|\bincident|\bvulnerability disclosure",
    "end of life": r"\bdecommission\w*|\bend of life\b|\bretir\w+",
}
LIFE_RE = {k: re.compile(v, re.I) for k, v in LIFECYCLE.items()}
OSS = re.compile(r"open[- ]source|\boss\b|open[- ]weights?|hugging ?face|\bsbom\b|model weights|third[- ]party", re.I)
OR_ELSE = re.compile(r"\bpenalt\w+|\bfines?\b|\bsanction\w*|\benforce\w*|\bliab\w+|\bprosecut\w+|\brevok\w+|"
                     r"\bloss of\b|\bexclu\w+ from\b|\bfailure to comply\b|\bnon-?complian\w+", re.I)
INCENTIVE = re.compile(r"\bdemonstrat\w+|\bcertif\w+|\beligib\w+|\badvantage\b|\bfunding\b", re.I)
COND = re.compile(r"^(before|after|during|when|whenever|where|if|unless|prior to|throughout|across|for|"
                  r"in order to|so that|once|until|alongside)\b", re.I)
PROBABILITY = re.compile(r"almost certainly|highly likely|\blikely\b|realistic possibility|\bunlikely\b", re.I)
POSSIBILITY = re.compile(r"\b(?:may|can|could|might) (?:also |still |even )?(?:be|have|include|lead|result|cause|"
                         r"become|occur|affect|increase|reduce|exploit|poison|use)\b", re.I)
BOILER = re.compile(r"cookie|javascript|share on|subscribe|copy link|download & print|back to top|"
                    r"was this (?:page|article) helpful|privacy notice|sign up", re.I)

nlp = spacy.load("en_core_web_sm")


def markers(s):
    hits, taken = [], []
    for rx, st, typ in MARKERS:
        for m in re.finditer(rx, s, re.I):
            if not any(a <= m.start() < b for a, b in taken):
                hits.append((m.group(0), st, typ)); taken.append((m.start(), m.end()))
    if SELF_COMMIT.search(s):
        hits.append(("will", 3, "self-commitment"))
    return hits


def actor_class(text):
    hits = [k for k, rx in ACTOR_RE.items() if rx.search(text or "")]
    for generic in ("organisations", "end users", "you (document audience)"):
        if len(hits) > 1 and generic in hits:
            hits.remove(generic)
    return "; ".join(hits)


def subtree(tok):
    return " ".join(t.text for t in tok.subtree)


def code_sentence(s, lead_in=""):
    ms = markers(s)
    inherited = False
    if not ms and lead_in and markers(lead_in):        # bullet under "Developers should:"
        ms, inherited = markers(lead_in), True
    top = max(ms, key=lambda m: m[1]) if ms else ("", 1, "descriptive")
    doc = nlp(s)
    attr, aim, obj, conds, notes = "", "", "", [], []
    anchor = None
    if top[0] and not inherited:
        first = top[0].split()[0].lower()
        anchor = next((t for t in doc if t.text.lower() == first), None)
    verb = None
    if anchor is not None and anchor.lemma_ in ("encourage", "urge", "invite", "recommend", "allow", "enable", "let", "help"):
        win = re.search(re.escape(anchor.text) + r"\s+(.{3,160}?)\s+to\s+\w+", s, re.I)
        if win and actor_class(win.group(1)):
            attr = win.group(1)
        verb = next((c for c in anchor.children if c.dep_ in ("xcomp", "ccomp")), anchor)
    elif anchor is not None and anchor.lemma_ in ("need", "have", "require", "expect"):
        verb = next((c for c in anchor.children if c.dep_ in ("xcomp", "ccomp")), anchor)
    elif anchor is not None:
        verb = anchor.head if anchor.dep_ in ("aux", "auxpass") or anchor.pos_ == "AUX" else anchor
    else:
        verb = next((t for t in doc if t.dep_ == "ROOT"), None)
    if verb is not None:
        aim = verb.lemma_
        kids = list(verb.children)
        passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in kids)
        agent = [c for c in kids if c.dep_ == "agent"]
        subj = [c for c in kids if c.dep_ in ("nsubj", "nsubjpass")]
        if not attr:
            if passive:
                attr = subtree(agent[0])[3:] if agent else ""
                obj = subtree(subj[0]) if subj else ""
                if not agent:
                    notes.append("passive, no agent")
            elif subj:
                attr = subtree(subj[0])
        if not obj:
            o = [c for c in kids if c.dep_ in ("dobj", "obj", "attr")]
            obj = subtree(o[0]) if o else ""
        conds = [subtree(c) for c in kids if c.dep_ in ("advcl", "prep") and COND.match(subtree(c))]
    if attr.lower() in ("it", "this", "that", "these", "they", "there", "which", "both documents"):
        notes.append(f"pronoun/document subject '{attr}'"); attr = ""
    if inherited:
        attr = actor_class(lead_in) or attr
        notes.append("deontic and actor inherited from bullet lead-in")
    ac = actor_class(attr)
    flags = []
    if PROBABILITY.search(s): flags.append("probability language")
    if top[2] == "permission/possibility" and POSSIBILITY.search(s): flags.append("possibility, not permission")
    if "attackers" in ac: flags.append("subject is an attacker")
    if re.search(r"\bgoals?\b|\boutcomes?\b", lead_in or "", re.I): flags.append("goal/outcome bullet")
    st = top[1]
    oe = "; ".join(sorted({m.group(0).lower() for m in OR_ELSE.finditer(s)}))
    if flags and any(f in ("probability language", "possibility, not permission", "subject is an attacker") for f in flags):
        stype = "probably exclude"
    elif top[2] == "self-commitment":
        stype = "self-assigned commitment"
    elif st == 1:
        stype = "descriptive: actors named, not obligated" if actor_class(s) else "descriptive"
    elif not ac:
        stype = "norm, unassigned"
    elif oe:
        stype = "rule"
    elif top[2] == "enabling":
        stype = "strategy"
    else:
        stype = "norm, weak" if st == 2 else "norm"
    life = [k for k, rx in LIFE_RE.items() if rx.search(s)]
    return dict(attribute=attr, actor_class=ac or ("UNSPECIFIED" if st > 1 else ""),
                deontic=top[0], strength=st, deontic_type=top[2], aim=aim, object=obj,
                conditions=" | ".join(conds), or_else=oe,
                actors_mentioned=actor_class(s),
                incentive="; ".join(sorted({m.group(0).lower() for m in INCENTIVE.finditer(s)})),
                statement_type=stype, lifecycle="; ".join(life),
                oss_terms="; ".join(sorted({m.group(0).lower() for m in OSS.finditer(s)})),
                flags="; ".join(flags), parse_notes="; ".join(notes))


def diagnose(doc_id, sents, stm):
    n = len(sents)
    deo = stm[stm.strength > 1]
    keep = deo[deo.statement_type != "probably exclude"]
    d = dict(doc_id=doc_id, sentences=n, screened=len(stm),
             with_deontic=len(deo), kept_deontic=len(keep),
             s4=int((keep.strength == 4).sum()), s3=int((keep.strength == 3).sum()), s2=int((keep.strength == 2).sum()),
             unassigned=int((keep.statement_type == "norm, unassigned").sum()),
             self_commitments=int((keep.statement_type == "self-assigned commitment").sum()),
             probably_exclude=int((deo.statement_type == "probably exclude").sum()),
             you_addressed=int(keep.actor_class.str.contains("you").sum()),
             goal_bullets=int(stm["flags"].str.contains("goal").sum()),
             oss_mentions=int((stm["oss_terms"] != "").sum()),
             boilerplate_leaks=int(sents.sentence.str.contains(BOILER).sum()),
             long_sentences=int((sents.sentence.str.split().str.len() > 70).sum()),
             fragments=int(((sents.sentence.str.split().str.len() < 4) & (sents.block_type != "bullet")).sum()),
             duplicates=int(sents.sentence.duplicated().sum()),
             bullets_no_lead_in=int(((sents.block_type == "bullet") & (sents.lead_in.fillna("") == "")).sum()))
    v = []
    if n == 0:
        v.append("NO TEXT: fetch or extraction failed, needs manual file or custom fetcher")
    else:
        if d["boilerplate_leaks"] / n > 0.03 or d["duplicates"] / n > 0.05:
            v.append("website text leaking in: needs a site-specific extraction rule")
        if (d["long_sentences"] + d["fragments"]) / n > 0.10:
            v.append("poor sentence splitting (tables, PDF layout or lists): needs custom parsing")
        if d["probably_exclude"] > max(5, 0.3 * max(d["with_deontic"], 1)):
            v.append("threat-assessment style text: screen over-captures, consider a custom rule or exclusion")
        if d["you_addressed"] > 0.3 * max(d["kept_deontic"], 1):
            v.append("mostly addressed to 'you': record the document audience")
        if d["goal_bullets"]:
            v.append("goal/outcome lists: needs a team decision rule (implied obligations)")
        if d["kept_deontic"] == 0:
            v.append("no obligations found: check it is in scope or the text was captured")
    d["verdict"] = " | ".join(v) or "generic OK"
    return d


def main(path):
    OUT.mkdir(exist_ok=True)
    sents = pd.read_csv(path, dtype={"sentence_no": int}).fillna("")
    rows = []
    for r in sents.itertuples():
        if not (markers(r.sentence) or ANY_ACTOR.search(r.sentence) or OSS.search(r.sentence)
                or (r.lead_in and markers(r.lead_in))):
            continue
        c = code_sentence(r.sentence, r.lead_in)
        rows.append(dict(doc_id=r.doc_id, issuer=r.issuer, no=r.sentence_no, section=r.section,
                         lead_in=r.lead_in, statement=r.sentence, **c))
    stm = pd.DataFrame(rows)
    stm.to_csv(OUT / "statements.csv", index=False)

    # edges like Srinidhi's example: issuer -> obligated actor, weighted by strength
    # strategies (enabling language) are kept in statements.csv but, as in the example, not drawn as edges
    obl = stm[(stm.strength > 1) & (~stm.statement_type.isin(["probably exclude", "strategy"]))].copy()
    obl["actor"] = obl.actor_class.str.split("; ")
    obl = obl.explode("actor")
    edges = (obl.groupby(["doc_id", "issuer", "actor"])
             .agg(statements=("no", "count"), strengths=("strength", lambda x: ", ".join(map(str, x))),
                  weight=("strength", "sum"),
                  self_loop=("statement_type", lambda x: (x == "self-assigned commitment").any()))
             .reset_index())
    edges.to_csv(OUT / "edges.csv", index=False)

    docs_csv = Path(path).with_name("documents.csv")
    doc_ids = pd.read_csv(docs_csv).doc_id.tolist() if docs_csv.exists() else list(sents.doc_id.unique())
    diag = pd.DataFrame([diagnose(d, sents[sents.doc_id == d], stm[stm.doc_id == d]) for d in doc_ids])
    if docs_csv.exists():
        diag = pd.read_csv(docs_csv)[["doc_id", "title", "status", "pages_fetched"]].merge(diag, on="doc_id")
    diag.to_csv(OUT / "diagnostics.csv", index=False)

    with pd.ExcelWriter(OUT / "first_pass.xlsx") as xw:
        diag.to_excel(xw, sheet_name="Diagnostics", index=False)
        stm.to_excel(xw, sheet_name="Statements", index=False)
        edges.to_excel(xw, sheet_name="Edges", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 90)
    print(diag[["doc_id", "sentences", "kept_deontic", "s4", "s3", "s2", "unassigned", "verdict"]].to_string(index=False))
    print(f"\n-> {OUT/'statements.csv'}, {OUT/'edges.csv'}, {OUT/'diagnostics.csv'}, {OUT/'first_pass.xlsx'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sentences.csv")
    main(ap.parse_args().input)
