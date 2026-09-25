"""Country-agnostic text normalisation for business names and addresses.

All dictionaries are hand-written rules (no external lookups). They cover US, India and French
conventions seen in the data; unknown countries fall through to the generic cleaning.
Non-Latin scripts (Devanagari, Kannada, ...) are transliterated to ASCII with unidecode.
"""
import re
import unicodedata

try:
    from unidecode import unidecode
except ImportError:  # fallback: strip accents only
    def unidecode(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

# ---------------------------------------------------------------- names
LEGAL = {
    # generic / US
    "inc": "inc", "incorporated": "inc", "corp": "corporation", "corporation": "corporation",
    "co": "company", "company": "company", "cos": "company", "llc": "llc", "llp": "llp", "lp": "lp",
    "ltd": "limited", "limited": "limited", "ltda": "limited", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "pa": "pa", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    # India
    "pvt": "private", "private": "private", "prv": "private", "opc": "opc", "huf": "huf",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "scop": "scop", "sca": "sca", "selarl": "selarl", "ets": "etablissements", "etablissements": "etablissements",
    "cie": "compagnie", "compagnie": "compagnie", "societe": "societe", "soc": "societe",
}
NAME_STOP = {"the", "and", "of", "a", "an", "et", "de", "du", "des", "la", "le", "les", "l", "d", "und",
             "mr", "mrs", "ms", "smt", "www", "com", "net", "org", "http", "https"}
NAME_WORD = {  # non-legal word abbreviations -> one canonical form
    "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "services",
    "tech": "technologies", "techs": "technologies", "technology": "technologies", "sys": "systems",
    "mgmt": "management", "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "ent": "enterprises", "entp": "enterprises", "ind": "industries", "inds": "industries", "grp": "group",
    "hldgs": "holdings", "natl": "national", "dept": "department", "univ": "university", "hosp": "hospital",
    "ctr": "center", "centre": "center", "mkt": "market", "pharma": "pharmaceuticals",
    "pharm": "pharmaceuticals", "engg": "engineering", "eng": "engineering", "dev": "development",
    "constr": "construction", "distr": "distributors", "saint": "st", "sainte": "ste", "freres": "brothers",
    "shree": "shri", "sri": "shri", "shreee": "shri", "projekts": "projects",
}
# "X dba Y", "X f/k/a Y", "X (Y)": second part is an alias / former name
ALIAS_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|trading as|t/a|aka|a\.k\.a\.?|"
                      r"f\s*/\s*k\s*/\s*a|fka|formerly known as|formerly)\b", re.I)
DOMAIN_RE = re.compile(r"\b([a-z0-9-]+)\.(?:com|in|net|org|co|fr|us|biz|info|co\.in)\b", re.I)

# ---------------------------------------------------------------- addresses
# canonical (short) street-type tokens; the same map is applied to both records, so any
# consistent choice works. "st" deliberately covers street AND saint (St.-Herblain / Saint-Herblain).
ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "place": "pl", "plaza": "plz", "highway": "hwy", "parkway": "pkwy", "square": "sq",
    "suite": "ste", "sainte": "ste", "apartment": "apt", "apts": "apt", "appt": "apt", "floor": "fl",
    "flr": "fl", "building": "bldg", "circle": "cir", "terrace": "ter", "trail": "trl", "mount": "mt",
    "fort": "ft", "route": "rte", "rt": "rte", "heights": "hts", "junction": "jct",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "room": "rm", "pob": "po",
    # India
    "marg": "rd", "ngr": "nagar", "nr": "near", "opposite": "opp", "society": "soc", "chs": "soc",
    "sector": "sec", "sect": "sec", "phase": "ph", "extension": "extn", "ext": "extn", "colony": "clny",
    "col": "clny", "market": "mkt", "station": "stn", "district": "dist", "distt": "dist",
    "taluka": "tal", "taluk": "tal", "tq": "tal", "village": "vill", "vil": "vill", "industrial": "indl",
    "estate": "est", "estt": "est", "complex": "cplx", "cmplx": "cplx", "bazar": "bazaar",
    # France
    "rue": "rue", "r": "rue", "chemin": "chem", "che": "chem", "impasse": "imp", "faubourg": "fbg",
    "fg": "fbg", "allee": "all", "allees": "all", "cours": "crs", "quai": "qu", "promenade": "prom",
    "residence": "res", "batiment": "bat", "lieudit": "ld",
}
ADDR_STOP = {"no", "nos", "number", "door", "dno", "hno", "hn", "house", "flat", "plot", "shop", "sno",
             "survey", "kh", "khasra", "null", "none", "na", "nil", "the", "of", "and", "de", "du", "des",
             "la", "le", "les", "l", "d", "a", "pin", "pincode", "cedex", "bis", "ter"}
STREET_WORDS = set(ADDR_ABBR.values()) | {"nagar", "near", "opp", "main", "cross", "layout", "block", "wing",
                                         "tower", "unit", "apt", "ste", "fl", "bldg", "rm", "po", "vill"}
CITY_ALIAS = {
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "benares": "varanasi", "banaras": "varanasi",
    "pondicherry": "puducherry", "allahabad": "prayagraj", "vizag": "visakhapatnam", "orissa": "odisha",
    "cuddapah": "kadapa",
}
# full name -> abbreviation (contracting is safe; expanding "in"/"or"/"de" would not be)
REGION_ABBR = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "nct of delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}
# French regions and departments: sources disagree on which one they give, so both are moved out of
# the address into a separate low-weight "region" field.
FR_REGIONS = [
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
    "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine", "occitanie",
    "pays de la loire", "provence alpes cote d azur", "provence alpes cote dazur",
    "ain", "aisne", "allier", "alpes de haute provence", "hautes alpes", "alpes maritimes", "ardeche",
    "ardennes", "ariege", "aube", "aude", "aveyron", "bouches du rhone", "calvados", "cantal", "charente",
    "charente maritime", "cher", "correze", "cote d or", "cotes d armor", "creuse", "dordogne", "doubs",
    "drome", "eure", "eure et loir", "finistere", "gard", "haute garonne", "gers", "gironde", "herault",
    "ille et vilaine", "indre", "indre et loire", "isere", "jura", "landes", "loir et cher", "loire",
    "haute loire", "loire atlantique", "loiret", "lot", "lot et garonne", "lozere", "maine et loire",
    "manche", "marne", "haute marne", "mayenne", "meurthe et moselle", "meuse", "morbihan", "moselle",
    "nievre", "nord", "oise", "orne", "pas de calais", "puy de dome", "pyrenees atlantiques",
    "hautes pyrenees", "pyrenees orientales", "bas rhin", "haut rhin", "rhone", "haute saone",
    "saone et loire", "sarthe", "savoie", "haute savoie", "seine maritime", "seine et marne", "yvelines",
    "deux sevres", "somme", "tarn", "tarn et garonne", "var", "vaucluse", "vendee", "vienne",
    "haute vienne", "vosges", "yonne", "territoire de belfort", "essonne", "hauts de seine",
    "seine saint denis", "seine st denis", "val de marne", "val d oise",
]
# Indian state names in native scripts, as they appear in some sources ("महाराष्ट्र", "ಕರ್ನಾಟಕ")
NATIVE_REGIONS = {
    "महाराष्ट्र": "mh", "उत्तर प्रदेश": "up", "कर्नाटक": "ka", "ಕರ್ನಾಟಕ": "ka", "तमिलनाडु": "tn",
    "தமிழ்நாடு": "tn", "तेलंगाना": "ts", "తెలంగాణ": "ts", "पश्चिम बंगाल": "wb", "পশ্চিমবঙ্গ": "wb",
    "गुजरात": "gj", "ગુજરાત": "gj", "दिल्ली": "dl", "राजस्थान": "rj", "हरियाणा": "hr", "केरल": "kl",
    "കേരളം": "kl", "आंध्र प्रदेश": "ap", "ఆంధ్ర ప్రదేశ్": "ap", "मध्य प्रदेश": "mp", "बिहार": "br",
    "पंजाब": "pb", "ਪੰਜਾਬ": "pb", "ओडिशा": "od", "ଓଡ଼ିଶା": "od", "झारखंड": "jh", "छत्तीसगढ़": "cg",
    "उत्तराखंड": "uk", "असम": "as", "গোৱা": "ga", "गोवा": "ga", "हिमाचल प्रदेश": "hp",
}
for _native, _abbr in NATIVE_REGIONS.items():
    REGION_ABBR.setdefault(" ".join(unidecode(_native).lower().split()), _abbr)
REGION_ABBR["delhi"] = "dl"
REGION_ABBR["new delhi"] = "dl"
_REGION_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, REGION_ABBR), key=len, reverse=True)) + r")\b")
# only remove a French region/department when it is a whole comma-separated component
_FR_SET = set(FR_REGIONS)
LANDMARK_RE = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj to|adj|close to|"
    r"in front of|facing|landmark|pres de|en face de|a cote de|derriere)\b\.?\s*([^,;]*)", re.I)
POSTCODE_PATTERNS = [
    re.compile(r"\b(\d{6})\b"),                 # India PIN
    re.compile(r"\b(\d{3})\s(\d{3})\b"),        # India PIN written "400 053"
    re.compile(r"\b(\d{5})(?:-\d{4})?\b"),      # US ZIP / FR code postal
]
_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def base_clean(s):
    s = unidecode(str(s or "")).lower()
    s = s.replace("&", " and ").replace("@", " at ").replace("+", " and ")
    s = _ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)  # l.l.c. -> llc, s.a.r.l. -> sarl
    s = s.replace("'", "")
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------- name
def _name_tokens(raw):
    raw = DOMAIN_RE.sub(lambda m: " " + m.group(1).replace("-", " ") + " ", str(raw or ""))
    return base_clean(raw).split()


_REPEAT_RE = re.compile(r"(.)\1+")
TRANSLIT_WORD = {"praivet": "private", "privet": "private", "praivhet": "private", "limited": "limited",
                 "limted": "limited", "kampani": "company", "kampni": "company", "endd": "and"}


def _is_translit(raw):
    """True when the name is written in a non-Latin script (Devanagari, Kannada, ...)."""
    return any(ord(c) > 0x24F and c.isalpha() for c in raw)


def norm_name(raw):
    """-> (name_norm, name_core, legal tokens, alias core)."""
    raw = str(raw or "")
    translit = _is_translit(raw)
    parts = ALIAS_RE.split(raw, maxsplit=1)
    alias_raw = parts[1] if len(parts) == 2 else ""
    if not alias_raw:
        m = re.search(r"[\(\[]([^\)\]]+)[\)\]]", raw)  # "Mohan (India) Technologies", "Mount Lutheran [Church]"
        alias_raw = m.group(1) if m else ""
    norm, core, legal = [], [], []
    for t in _name_tokens(ALIAS_RE.sub(" ", raw)):
        if translit:  # "praaivett limittedd" -> "praivet limited"
            t = _REPEAT_RE.sub(r"\1", t)
            t = TRANSLIT_WORD.get(t, t)
        t = NAME_WORD.get(t, t)
        if t in LEGAL:
            legal.append(LEGAL[t])
            norm.append(LEGAL[t])
            continue
        norm.append(t)
        if t not in NAME_STOP:
            core.append(t)
    if not core:  # name was only legal words
        core = [t for t in norm if t not in NAME_STOP] or norm
    alias = [NAME_WORD.get(t, t) for t in _name_tokens(alias_raw)]
    alias = [t for t in alias if t not in LEGAL and t not in NAME_STOP]
    return " ".join(norm), " ".join(core), " ".join(sorted(set(legal))), " ".join(alias)


def acronym(core):
    toks = core.split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


# ---------------------------------------------------------------- address
def extract_postcode(clean_addr):
    """Only a 5/6-digit number (or 'ddd ddd') at the very END counts: elsewhere it is a house number."""
    m = re.search(r"\b(\d{3})\s?(\d{3})$|\b(\d{5})(?:\s\d{4})?$", clean_addr)
    if not m:
        return ""
    return m.group(3) or (m.group(1) + m.group(2))


_NUM_PREFIX_RE = re.compile(r"\b(sno|hno|dno|no|plot|flat|shop|door|house)(\d)")


def norm_addr(raw):
    """-> (addr_core, postcode, number tokens, landmark, region)."""
    raw = unidecode(str(raw or ""))
    landmarks = [m.group(1) for m in LANDMARK_RE.finditer(raw)]
    raw = LANDMARK_RE.sub(" ", raw)
    region, keep = [], []
    for comp in raw.split(","):
        c = base_clean(comp)
        if c in _FR_SET:
            region.append(c)
        elif c:
            keep.append(c)
    clean = " ".join(keep)
    postcode = extract_postcode(clean)
    if postcode:
        clean = clean[: re.search(r"\b\d{3}\s?\d{3}$|\b\d{5}(?:\s\d{4})?$", clean).start()]
    clean = _REGION_RE.sub(lambda m: REGION_ABBR[m.group(1)], clean)
    toks = []
    for t in _NUM_PREFIX_RE.sub(r"\2", clean).split():
        if t.isdigit():
            t = t.lstrip("0") or "0"  # 00272 -> 272
        t = ADDR_ABBR.get(t, t)
        t = CITY_ALIAS.get(t, t)
        if t not in ADDR_STOP:
            toks.append(t)
    numbers = [t for t in toks if any(c.isdigit() for c in t)]
    lm = " ".join(base_clean(x) for x in landmarks if x.strip())
    return " ".join(toks), postcode, " ".join(dict.fromkeys(numbers)), lm, " ".join(region)


def norm_record(name, addr):
    nn, nc, nl, na = norm_name(name)
    ac, pc, nums, lm, reg = norm_addr(addr)
    return nn, nc, nl, na, acronym(nc), ac, pc, nums, lm, reg


NORM_FIELDS = ["name_norm", "name_core", "name_legal", "name_alias", "name_acr",
               "addr_core", "postcode", "addr_nums", "landmark", "region"]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        ("Kochar Góld Private", "Maharashtra, NO D/1A 1002, ARKADE ART COMPLEX, THANE"),
        ("श्री प्रोजेक्ट्स प्राइवेट लिमिटेड", "SNO175/A  VAIIABHANGR SWAR, PUNE, PUNE CITY, Maharashtra"),
        ("Halotavo F/K/A Roach Beverage Corp", "00272 LAGO GRANDE DRIVE, HORIZON CITY, TX"),
        ("victorylaboratories.com", "B-61 Sector-2, Gautam Buddha Nagar, Noida, उत्तर प्रदेश"),
        ("Saint-Herblain Sportif SARL", "30 RUE BLANDINE, ST.-HERBLAIN, Pays de la Loire"),
        ("Utilisateur Compagnie Holding", "Nº 14 Pl De Labbe Bonpain, Dunkerque"),
        ("Mount Lutheran [Church]", "340 Almond St, # 4, Fall River, Massachusetts"),
    ]
    for n, a in tests:
        print(n, "|", a, "\n   ", norm_record(n, a))
