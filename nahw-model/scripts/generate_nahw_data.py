# -*- coding: utf-8 -*-
"""
Generates a labeled Arabic nahw (grammar) dataset by substituting words into
hand-checked, fully-diacritized template sentences. Output format matches
lang-model's corpus format:  word<TAB>TAG   (blank line = new sentence)

IMPORTANT LIMITATION (read this before trusting the tagger):
Attached one-letter prefixes (و العطف / ب الجر / ل الجر / ك الجر) are NOT
separated by whitespace in real Arabic ("والمعلمُ" is one surface token that
is really two morphemes). This generator therefore avoids attached-prefix
patterns and only uses free-standing prepositions/conjunctions (إلى، من، في،
على، مع، ثم) so every tag lines up with a real whitespace-separated token.
A model trained on this data will mislabel attached-prefix words if you feed
it real, unfiltered text - that needs morphological segmentation first,
which this project does not do.

Run:  python scripts/generate_nahw_data.py --out rdf/nahw_data.txt
"""
import argparse
import itertools
import random

# ---------------------------------------------------------------- word bank
# Each noun given as (definite-raf3, definite-nasb, definite-jarr,
#                     indef-raf3,   indef-nasb,    indef-jarr)
NOUNS = [
    ("الطالبُ", "الطالبَ", "الطالبِ", "طالبٌ", "طالبًا", "طالبٍ"),
    ("المعلمُ", "المعلمَ", "المعلمِ", "معلمٌ", "معلمًا", "معلمٍ"),
    ("الولدُ", "الولدَ", "الولدِ", "ولدٌ", "ولدًا", "ولدٍ"),
    ("الرجلُ", "الرجلَ", "الرجلِ", "رجلٌ", "رجلًا", "رجلٍ"),
    ("الكتابُ", "الكتابَ", "الكتابِ", "كتابٌ", "كتابًا", "كتابٍ"),
    ("القلمُ", "القلمَ", "القلمِ", "قلمٌ", "قلمًا", "قلمٍ"),
    ("البابُ", "البابَ", "البابِ", "بابٌ", "بابًا", "بابٍ"),
    ("البيتُ", "البيتَ", "البيتِ", "بيتٌ", "بيتًا", "بيتٍ"),
    ("المدرسةُ", "المدرسةَ", "المدرسةِ", "مدرسةٌ", "مدرسةً", "مدرسةٍ"),
    ("الحديقةُ", "الحديقةَ", "الحديقةِ", "حديقةٌ", "حديقةً", "حديقةٍ"),
    ("السوقُ", "السوقَ", "السوقِ", "سوقٌ", "سوقًا", "سوقٍ"),
    ("الكوبُ", "الكوبَ", "الكوبِ", "كوبٌ", "كوبًا", "كوبٍ"),
]
# adjectives (indef-raf3, indef-nasb, indef-jarr) - used as نعت / خبر / حال
ADJ = [
    ("مجتهدٌ", "مجتهدًا", "مجتهدٍ"),
    ("جميلٌ", "جميلًا", "جميلٍ"),
    ("كبيرٌ", "كبيرًا", "كبيرٍ"),
    ("صغيرٌ", "صغيرًا", "صغيرٍ"),
    ("جديدٌ", "جديدًا", "جديدٍ"),
    ("نظيفٌ", "نظيفًا", "نظيفٍ"),
]
HAAL_WORDS = ["مسرعًا", "ضاحكًا", "مسرورًا", "باكيًا", "خائفًا"]
PAST_VERBS = ["ذهبَ", "قرأَ", "كتبَ", "شربَ", "لعبَ", "جلسَ", "خرجَ", "دخلَ"]
# (present-marfu3, present-mansub-after-lan, present-majzum-after-lam)
PRESENT_VERBS = [
    ("يذهبُ", "يذهبَ", "يذهبْ"),
    ("يقرأُ", "يقرأَ", "يقرأْ"),
    ("يكتبُ", "يكتبَ", "يكتبْ"),
    ("يلعبُ", "يلعبَ", "يلعبْ"),
]
PREPOSITIONS = ["إلى", "من", "في", "على", "مع"]
TAMYIZ_PAIRS = [("امتلأ", "الكوبُ", "ماءً"), ("امتلأت", "الحديقةُ", "وردًا")]


def sent(pairs):
    """pairs: list of (word, tag) -> one training sentence (list of lines)."""
    return [f"{w}\t{t}" for w, t in pairs] + [""]


def gen_ismiyya_basic(rng, n):
    """مبتدأ + خبر: الطالبُ مجتهدٌ."""
    out = []
    for noun, adj in itertools.islice(
            ((rng.choice(NOUNS), rng.choice(ADJ)) for _ in itertools.count()), n):
        out += sent([(noun[0], "مبتدأ"), (adj[0], "خبر")])
    return out


def gen_fi3liyya_fail(rng, n):
    """فعل ماض + فاعل: ذهبَ الطالبُ."""
    out = []
    for _ in range(n):
        v, noun = rng.choice(PAST_VERBS), rng.choice(NOUNS)
        out += sent([(v, "فعل_ماض"), (noun[0], "فاعل")])
    return out


def gen_fail_maful(rng, n):
    """فعل + فاعل + مفعول به: قرأَ الطالبُ الكتابَ."""
    out = []
    for _ in range(n):
        v = rng.choice(PAST_VERBS)
        subj, obj = rng.sample(NOUNS, 2)
        out += sent([(v, "فعل_ماض"), (subj[0], "فاعل"), (obj[1], "مفعول_به")])
    return out


def gen_idhafa(rng, n):
    """مضاف + مضاف إليه + خبر: كتابُ الطالبِ جميلٌ."""
    out = []
    for _ in range(n):
        mudaf, mudaf_ilayh = rng.sample(NOUNS, 2)
        adj = rng.choice(ADJ)
        out += sent([
            (mudaf[0].replace("ال", "", 1), "مضاف"),   # مضاف (no ال, keeps case)
            (mudaf_ilayh[2], "مضاف_إليه"),
            (adj[0], "خبر"),
        ])
    return out


def gen_jar_majrur(rng, n):
    """فعل + فاعل + حرف جر + اسم مجرور: ذهبَ الطالبُ إلى المدرسةِ."""
    out = []
    for _ in range(n):
        v, noun = rng.choice(PAST_VERBS), rng.choice(NOUNS)
        prep = rng.choice(PREPOSITIONS)
        dest = rng.choice(NOUNS)
        out += sent([(v, "فعل_ماض"), (noun[0], "فاعل"),
                     (prep, "حرف_جر"), (dest[2], "اسم_مجرور")])
    return out


def gen_naat(rng, n):
    """فعل + فاعل + مفعول به + نعت: رأى الطالبُ كتابًا جديدًا."""
    out = []
    for _ in range(n):
        subj, obj = rng.sample(NOUNS, 2)
        adj = rng.choice(ADJ)
        out += sent([("رأى", "فعل_ماض"), (subj[0], "فاعل"),
                     (obj[4], "مفعول_به"), (adj[1], "نعت_منصوب")])
    return out


def gen_haal(rng, n):
    """فعل + فاعل + حال: جاءَ الطالبُ مسرعًا."""
    out = []
    for _ in range(n):
        noun = rng.choice(NOUNS)
        haal = rng.choice(HAAL_WORDS)
        out += sent([("جاءَ", "فعل_ماض"), (noun[0], "فاعل"), (haal, "حال")])
    return out


def gen_tamyiz(rng, n):
    """امتلأ + فاعل + تمييز: امتلأ الكوبُ ماءً."""
    out = []
    for _ in range(n):
        v, subj, tamyiz = rng.choice(TAMYIZ_PAIRS)
        out += sent([(v, "فعل_ماض"), (subj, "فاعل"), (tamyiz, "تمييز")])
    return out


def gen_zarf(rng, n):
    """فعل + فاعل + ظرف (مفعول فيه): لعبَ الولدُ صباحًا."""
    out = []
    zarfs = ["صباحًا", "مساءً", "ليلًا", "نهارًا"]
    for _ in range(n):
        v, noun, z = rng.choice(PAST_VERBS), rng.choice(NOUNS), rng.choice(zarfs)
        out += sent([(v, "فعل_ماض"), (noun[0], "فاعل"), (z, "مفعول_فيه_ظرف")])
    return out


def gen_maful_lah(rng, n):
    """فعل + فاعل + مفعول لأجله: سافرَ الرجلُ طلبًا."""
    out = []
    reasons = ["طلبًا", "رغبةً", "خوفًا", "حبًا"]
    for _ in range(n):
        noun = rng.choice(NOUNS)
        reason = rng.choice(reasons)
        out += sent([("سافرَ", "فعل_ماض"), (noun[0], "فاعل"), (reason, "مفعول_لأجله")])
    return out


def gen_atf(rng, n):
    """فعل + فاعل + حرف عطف + معطوف: نجحَ الطالبُ ثم المعلمُ."""
    out = []
    for _ in range(n):
        subj, atf = rng.sample(NOUNS, 2)
        out += sent([("نجحَ", "فعل_ماض"), (subj[0], "فاعل"),
                     ("ثم", "حرف_عطف"), (atf[0], "معطوف_مرفوع")])
    return out


def gen_tawkid(rng, n):
    """فعل + فاعل + توكيد: حضرَ الطلابُ جميعُهم. (fixed forms, plural)"""
    out = []
    subjects = ["الطلابُ", "المعلمونَ", "الأولادُ"]
    for _ in range(n):
        subj = rng.choice(subjects)
        out += sent([("حضرَ", "فعل_ماض"), (subj, "فاعل"), ("جميعُهم", "توكيد_مرفوع")])
    return out


def gen_mudari_marfu3(rng, n):
    """فعل مضارع مرفوع (افتراضي) + فاعل + جار ومجرور."""
    out = []
    for _ in range(n):
        v_forms = rng.choice(PRESENT_VERBS)
        noun = rng.choice(NOUNS)
        prep, dest = rng.choice(PREPOSITIONS), rng.choice(NOUNS)
        out += sent([(v_forms[0], "فعل_مضارع_مرفوع"), (noun[0], "فاعل"),
                     (prep, "حرف_جر"), (dest[2], "اسم_مجرور")])
    return out


def gen_mudari_mansub(rng, n):
    """لن + فعل مضارع منصوب + فاعل: لن يذهبَ الطالبُ."""
    out = []
    for _ in range(n):
        v_forms = rng.choice(PRESENT_VERBS)
        noun = rng.choice(NOUNS)
        out += sent([("لن", "حرف_نصب"), (v_forms[1], "فعل_مضارع_منصوب"), (noun[0], "فاعل")])
    return out


def gen_mudari_majzum(rng, n):
    """لم + فعل مضارع مجزوم + فاعل: لم يذهبْ الولدُ."""
    out = []
    for _ in range(n):
        v_forms = rng.choice(PRESENT_VERBS)
        noun = rng.choice(NOUNS)
        out += sent([("لم", "حرف_نفي_جازم"), (v_forms[2], "فعل_مضارع_مجزوم"), (noun[0], "فاعل")])
    return out


GENERATORS = [
    gen_ismiyya_basic, gen_fi3liyya_fail, gen_fail_maful, gen_idhafa,
    gen_jar_majrur, gen_naat, gen_haal, gen_tamyiz, gen_zarf,
    gen_maful_lah, gen_atf, gen_tawkid,
    gen_mudari_marfu3, gen_mudari_mansub, gen_mudari_majzum,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rdf/nahw_data.txt")
    ap.add_argument("--per-pattern", type=int, default=120,
                    help="how many example sentences to generate per template pattern")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lines, seen = [], set()
    for gen in GENERATORS:
        block = gen(rng, args.per_pattern * 3)  # over-generate, dedupe below
        # split back into sentences on blank lines, dedupe identical sentences
        cur, kept = [], []
        for ln in block:
            if ln == "":
                key = tuple(cur)
                if cur and key not in seen:
                    seen.add(key)
                    kept.append(cur)
                cur = []
            else:
                cur.append(ln)
        rng.shuffle(kept)
        for s in kept[:args.per_pattern]:
            lines += s + [""]
        print(f"{gen.__name__:<22} -> {min(len(kept), args.per_pattern)} unique sentences")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    n_sents = sum(1 for ln in lines if ln == "")
    print(f"\nTotal sentences: {n_sents}  -> {args.out}")


if __name__ == "__main__":
    main()
