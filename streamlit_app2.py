import json
import os
import re
import calendar
from datetime import date, timedelta, datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ============================================================
# ARCHIVES / SAUVEGARDE / RÉVISION / INDU-DÛ
# ============================================================

SAVE_DIR = "saved_cases"

def _ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)

def _now_iso():
    return datetime.now().isoformat(timespec="seconds")

def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:80] if s else "sans_nom"

def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return str(o)

def save_case(case_payload: dict) -> str:
    """
    Sauve un calcul complet (answers + résultats + meta) dans un JSON.
    Retourne le case_id.
    """
    _ensure_save_dir()
    case_id = case_payload.get("meta", {}).get("case_id") or f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    case_payload.setdefault("meta", {})
    case_payload["meta"]["case_id"] = case_id
    case_payload["meta"]["saved_at"] = _now_iso()

    demandeur = _safe_filename(case_payload["meta"].get("demandeur_nom", "") or "sans_nom")
    fname = f"{case_id}__{demandeur}.json"
    path = os.path.join(SAVE_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case_payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return case_id

def list_cases() -> list:
    """
    Retourne une liste de dict {path, meta, payload} triée par date desc.
    """
    _ensure_save_dir()
    items = []
    for fn in os.listdir(SAVE_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(SAVE_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            meta = payload.get("meta", {})
            items.append({"path": path, "meta": meta, "payload": payload})
        except Exception:
            continue

    def key(it):
        return it.get("meta", {}).get("saved_at") or it.get("meta", {}).get("created_at") or ""
    items.sort(key=key, reverse=True)
    return items

def load_case_by_path(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# CONFIG PAR DÉFAUT (fusion avec ris_rules.json si présent)
# ============================================================
DEFAULT_ENGINE = {
    "version": "1.7",
    "config": {
        "ris_rates_annuel": {"cohab": 10513.60, "isole": 15770.41, "fam_charge": 21312.87},
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},
        "art34": {"taux_a_laisser_mensuel": 876.13},
        "pf": {"pf_mensuel_defaut": 240.0},
        "capital_mobilier": {
            "t0_max": 6199.0,
            "t1_min": 6200.0,
            "t1_max": 12500.0,
            "t1_rate": 0.06,
            "t2_rate": 0.10
        },
        "immo": {
            "bati_base": 750.0,
            "bati_par_enfant": 125.0,
            "non_bati_base": 30.0,
            "coeff_rc": 3.0
        },
        "socio_prof": {
            "max_mensuel": 309.48,
            "artistique_annuel": 3297.80,
        },
        "cession": {
            "tranche_immunisee": 37200.0,
            "usufruit_ratio": 0.40,
            "abattements_annuels": {"cat1": 1250.0, "cat2": 2000.0, "cat3": 2500.0}
        },
        "ale": {"valeur_cheque": 4.35, "exon_par_cheque": 6.0}
    }
}

# ============================================================
# UTILITAIRES
# ============================================================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def r2(x: float) -> float:
    return float(round(float(x), 2))

def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def normalize_engine(raw: dict) -> dict:
    raw = raw or {}
    engine = deep_merge(DEFAULT_ENGINE, raw)
    cfg = engine["config"]

    if "exonerations" in cfg and "immo" in cfg:
        exo = cfg["exonerations"]
        cfg["immo"]["bati_base"] = float(exo.get("bati_base", cfg["immo"]["bati_base"]))
        cfg["immo"]["bati_par_enfant"] = float(exo.get("bati_par_enfant", cfg["immo"]["bati_par_enfant"]))
        cfg["immo"]["non_bati_base"] = float(exo.get("non_bati_base", cfg["immo"]["non_bati_base"]))

    if "ris_rates_annuel" not in cfg:
        cfg["ris_rates_annuel"] = {"cohab": 0.0, "isole": 0.0, "fam_charge": 0.0}

    for k in ("cohab", "isole", "fam_charge"):
        cfg["ris_rates"][k] = float(cfg["ris_rates"].get(k, 0.0))
        cfg["immunisation_simple_annuelle"][k] = float(cfg["immunisation_simple_annuelle"].get(k, 0.0))
        cfg["ris_rates_annuel"][k] = float(cfg["ris_rates_annuel"].get(k, 0.0) or 0.0)

    if "art34" not in cfg:
        cfg["art34"] = {}
    cfg["art34"]["taux_a_laisser_mensuel"] = float(
        cfg["art34"].get("taux_a_laisser_mensuel", cfg["ris_rates"].get("cohab", 0.0))
    )

    if "pf" not in cfg:
        cfg["pf"] = {"pf_mensuel_defaut": 0.0}
    cfg["pf"]["pf_mensuel_defaut"] = float(cfg["pf"].get("pf_mensuel_defaut", 0.0))

    if "ale" not in cfg:
        cfg["ale"] = {"valeur_cheque": 0.0, "exon_par_cheque": 6.0}
    cfg["ale"]["valeur_cheque"] = float(cfg["ale"].get("valeur_cheque", 0.0))
    cfg["ale"]["exon_par_cheque"] = float(cfg["ale"].get("exon_par_cheque", 6.0))

    return engine

def load_engine() -> dict:
    if os.path.exists("ris_rules.json"):
        with open("ris_rules.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        return normalize_engine(raw)
    return normalize_engine(DEFAULT_ENGINE)

def end_of_month(d: date) -> date:
    dim = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, dim)

def next_day(d: date) -> date:
    return d + timedelta(days=1)

def date_in_same_month(d: date, ref: date) -> bool:
    return d.year == ref.year and d.month == ref.month

def safe_parse_date(x):
    if isinstance(x, date):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return date.fromisoformat(x.strip())
        except Exception:
            return None
    return None

def euro(x: float) -> str:
    x = float(x or 0.0)
    s = f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def cat_label(cat: str) -> str:
    cat = (cat or "").strip().lower()
    mapping = {"cohab": "Cohabitant", "isole": "Isolé", "fam_charge": "Famille à charge"}
    return mapping.get(cat, cat)

# ============================================================
# PATRIMOINE — commun + perso (mêmes clés)
# ============================================================
PATRIMOINES_KEYS = {
    "capital_mobilier_total", "capital_compte_commun", "capital_nb_titulaires",
    "capital_conjoint_cotitulaire", "capital_fraction",
    "biens_immobiliers",
    "cessions", "cession_cas_particulier_37200", "cession_dettes_deductibles",
    "cession_abatt_cat", "cession_abatt_mois",
    "avantage_nature_logement_mensuel",
}

def _pat_default() -> dict:
    return {
        "capital_mobilier_total": 0.0,
        "capital_compte_commun": False,
        "capital_nb_titulaires": 1,
        "capital_conjoint_cotitulaire": False,
        "capital_fraction": 1.0,
        "biens_immobiliers": [],
        "cessions": [],
        "cession_cas_particulier_37200": False,
        "cession_dettes_deductibles": 0.0,
        "cession_abatt_cat": "cat1",
        "cession_abatt_mois": 0,
        "avantage_nature_logement_mensuel": 0.0,
    }

def _extract_patrimoine(d: Optional[dict]) -> dict:
    base = _pat_default()
    d = d or {}
    for k in PATRIMOINES_KEYS:
        if k in d:
            base[k] = d[k]
    base["biens_immobiliers"] = list(base.get("biens_immobiliers") or [])
    base["cessions"] = list(base.get("cessions") or [])
    base["capital_mobilier_total"] = float(base.get("capital_mobilier_total") or 0.0)
    base["avantage_nature_logement_mensuel"] = float(base.get("avantage_nature_logement_mensuel") or 0.0)
    base["capital_nb_titulaires"] = max(1, int(base.get("capital_nb_titulaires") or 1))
    base["capital_fraction"] = clamp01(float(base.get("capital_fraction") or 1.0))
    return base

# ============================================================
# CAPITAUX MOBILIERS (annuel) - détail tranches
# ============================================================
def capital_mobilier_calc(total_capital: float,
                          compte_commun: bool,
                          nb_titulaires: int,
                          categorie: str,
                          conjoint_compte_commun: bool,
                          part_fraction_custom: float,
                          cfg_cap: dict) -> dict:
    total_capital = max(0.0, float(total_capital))

    if compte_commun:
        nb = max(1, int(nb_titulaires))
        numerator = 2 if (categorie == "fam_charge" and conjoint_compte_commun) else 1
        fraction = numerator / nb
        fraction_mode = f"compte commun: {numerator}/{nb}"
    else:
        fraction = clamp01(part_fraction_custom)
        fraction_mode = f"fraction custom: {fraction:.2f}"

    adj_total = total_capital * fraction

    t0_max = float(cfg_cap["t0_max"]) * fraction
    t1_min = float(cfg_cap["t1_min"]) * fraction
    t1_max = float(cfg_cap["t1_max"]) * fraction
    r1_ = float(cfg_cap["t1_rate"])
    r2_ = float(cfg_cap["t2_rate"])

    tranche1_base = max(0.0, min(adj_total, t1_max) - t1_min)
    tranche2_base = max(0.0, adj_total - t1_max)

    tranche1_calc = tranche1_base * r1_
    tranche2_calc = tranche2_base * r2_

    annuel = 0.0 if adj_total <= t0_max else (tranche1_calc + tranche2_calc)

    return {
        "total_capital": r2(total_capital),
        "fraction": r2(fraction),
        "fraction_mode": fraction_mode,
        "capital_pris_en_compte_base": r2(adj_total),
        "seuils": {
            "t0_max": r2(t0_max),
            "t1_min": r2(t1_min),
            "t1_max": r2(t1_max),
            "t1_rate": r2(r1_),
            "t2_rate": r2(r2_),
        },
        "tranches": [
            {"label": "Tranche 1", "base": r2(tranche1_base), "taux": r2(r1_), "produit": r2(tranche1_calc),
             "borne": f"]{r2(t1_min)} ; {r2(t1_max)}]"},
            {"label": "Tranche 2", "base": r2(tranche2_base), "taux": r2(r2_), "produit": r2(tranche2_calc),
             "borne": f">{r2(t1_max)}"},
        ],
        "annuel": r2(max(0.0, annuel))
    }

# ============================================================
# IMMOBILIER (annuel) - détail
# ============================================================
def immo_calc_total(biens: list, enfants: int, cfg_immo: dict) -> dict:
    biens_countes = [b for b in biens if not b.get("habitation_principale", False)]
    nb_bati = sum(1 for b in biens_countes if b.get("bati", True))
    nb_non_bati = sum(1 for b in biens_countes if not b.get("bati", True))

    exo_bati_total = float(cfg_immo["bati_base"]) + float(cfg_immo["bati_par_enfant"]) * max(0, int(enfants))
    exo_non_bati_total = float(cfg_immo["non_bati_base"])
    coeff = float(cfg_immo.get("coeff_rc", 3.0))

    details = []
    total_annuel = 0.0

    for idx, b in enumerate(biens_countes, start=1):
        bati = bool(b.get("bati", True))
        rc = max(0.0, float(b.get("rc_non_indexe", 0.0)))
        frac = clamp01(float(b.get("fraction_droits", 1.0)))
        rc_part = rc * frac

        if bati:
            exo_par_bien = (exo_bati_total * frac) / nb_bati if nb_bati > 0 else 0.0
            typ = "Bâti"
        else:
            exo_par_bien = (exo_non_bati_total * frac) / nb_non_bati if nb_non_bati > 0 else 0.0
            typ = "Non bâti"

        base_rc = max(0.0, rc_part - exo_par_bien)
        base_coeff = base_rc * coeff

        ded_interets = 0.0
        ded_rente = 0.0

        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            ded_interets = min(interets, 0.5 * base_coeff)
            base_coeff -= ded_interets

        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            ded_rente = min(rente, 0.5 * base_coeff)
            base_coeff -= ded_rente

        pris = max(0.0, base_coeff)
        total_annuel += pris

        details.append({
            "bien": idx,
            "type": typ,
            "rc_non_indexe": r2(rc),
            "fraction": r2(frac),
            "rc_part": r2(rc_part),
            "exo_par_bien": r2(exo_par_bien),
            "rc_apres_exo": r2(base_rc),
            "coeff_rc": r2(coeff),
            "base_coeff": r2(base_rc * coeff),
            "ded_interets": r2(ded_interets),
            "ded_rente": r2(ded_rente),
            "pris_en_compte": r2(pris),
        })

    return {
        "total_annuel": r2(max(0.0, total_annuel)),
        "coeff_rc": r2(coeff),
        "exo_bati_total": r2(exo_bati_total),
        "exo_non_bati_total": r2(exo_non_bati_total),
        "nb_bati": int(nb_bati),
        "nb_non_bati": int(nb_non_bati),
        "details": details,
    }

# ============================================================
# CESSION DE BIENS (annuel) - détail
# ============================================================
def cession_biens_calc(cessions: list,
                       cas_particulier_tranche_37200: bool,
                       dettes_deductibles: float,
                       abatt_cat: str,
                       abatt_mois_prorata: int,
                       cfg_cession: dict,
                       cfg_cap: dict) -> dict:
    brut = 0.0
    details_cess = []
    for i, c in enumerate(cessions or [], start=1):
        v = max(0.0, float(c.get("valeur_venale", 0.0)))
        v0 = v
        if c.get("usufruit", False):
            v = v * float(cfg_cession["usufruit_ratio"])
        brut += v
        details_cess.append({
            "cession": i,
            "valeur_venale": r2(v0),
            "usufruit": bool(c.get("usufruit", False)),
            "ratio_usufruit": r2(float(cfg_cession["usufruit_ratio"])),
            "valeur_retendue": r2(v),
        })

    dettes = max(0.0, float(dettes_deductibles))
    apres_dettes = max(0.0, brut - dettes)

    tranche_immunisee = float(cfg_cession["tranche_immunisee"]) if cas_particulier_tranche_37200 else 0.0
    apres_tranche = max(0.0, apres_dettes - tranche_immunisee)

    abatt_annuel = float(cfg_cession["abattements_annuels"].get(abatt_cat, 0.0))
    mois = max(0, min(12, int(abatt_mois_prorata)))
    abatt_prorata = abatt_annuel * (mois / 12.0)
    base_cap = max(0.0, apres_tranche - abatt_prorata)

    t0_max = float(cfg_cap["t0_max"])
    t1_min = float(cfg_cap["t1_min"])
    t1_max = float(cfg_cap["t1_max"])
    r1_ = float(cfg_cap["t1_rate"])
    r2_ = float(cfg_cap["t2_rate"])

    tranche1_base = max(0.0, min(base_cap, t1_max) - t1_min)
    tranche2_base = max(0.0, base_cap - t1_max)
    tranche1_calc = tranche1_base * r1_
    tranche2_calc = tranche2_base * r2_

    annuel = 0.0 if base_cap <= t0_max else (tranche1_calc + tranche2_calc)

    return {
        "brut_total": r2(brut),
        "details_cessions": details_cess,
        "dettes_deductibles": r2(dettes),
        "apres_dettes": r2(apres_dettes),
        "cas_tranche_37200": bool(cas_particulier_tranche_37200),
        "tranche_37200": r2(tranche_immunisee),
        "apres_tranche_37200": r2(apres_tranche),
        "abatt_cat": abatt_cat,
        "abatt_annuel": r2(abatt_annuel),
        "abatt_mois": int(mois),
        "abatt_prorata": r2(abatt_prorata),
        "base_cap": r2(base_cap),
        "seuils": {"t0_max": r2(t0_max), "t1_min": r2(t1_min), "t1_max": r2(t1_max), "t1_rate": r2(r1_), "t2_rate": r2(r2_)},
        "tranches": [
            {"label": "Tranche 1", "base": r2(tranche1_base), "taux": r2(r1_), "produit": r2(tranche1_calc), "borne": f"]{r2(t1_min)} ; {r2(t1_max)}]"},
            {"label": "Tranche 2", "base": r2(tranche2_base), "taux": r2(r2_), "produit": r2(tranche2_calc), "borne": f">{r2(t1_max)}"},
        ],
        "annuel": r2(max(0.0, annuel)),
    }

# ============================================================
# REVENUS + ALE
# ============================================================
def _ale_montants(nb_cheques_mois: float, cfg_ale: dict) -> Tuple[float, float, float]:
    nb = max(0.0, float(nb_cheques_mois))
    val = max(0.0, float(cfg_ale.get("valeur_cheque", 0.0)))
    exo = max(0.0, float(cfg_ale.get("exon_par_cheque", 6.0)))
    brut_m = nb * val
    exo_m = nb * exo
    a_compter_m = max(0.0, brut_m - exo_m)
    return r2(brut_m), r2(exo_m), r2(a_compter_m)

def revenus_annuels_apres_exonerations(revenus_annuels: list, cfg_soc: dict, cfg_ale: dict) -> float:
    total_m = 0.0
    for r in revenus_annuels or []:
        t = (r.get("type", "standard") or "standard")
        eligible = bool(r.get("eligible", True))

        if t == "ale":
            if "nb_cheques_mois" in r:
                _brut_m, _exo_m, a_compter_m = _ale_montants(r.get("nb_cheques_mois", 0), cfg_ale)
                total_m += a_compter_m
            else:
                total_m += max(0.0, float(r.get("ale_part_excedentaire_mensuel", 0.0)))
            continue

        a = max(0.0, float(r.get("montant_annuel", 0.0)))
        m = a / 12.0

        if t in ("socio_prof", "etudiant") and eligible:
            ded = min(float(cfg_soc["max_mensuel"]), m)
            total_m += max(0.0, m - ded)
        elif t == "artistique_irregulier" and eligible:
            ded_m = float(cfg_soc["artistique_annuel"]) / 12.0
            total_m += max(0.0, m - min(ded_m, m))
        else:
            total_m += m

    return float(max(0.0, total_m * 12.0))

def annual_from_revenus_list(rev_list: list, cfg_soc: dict, cfg_ale: dict) -> float:
    return float(revenus_annuels_apres_exonerations(rev_list or [], cfg_soc, cfg_ale))

# ============================================================
# ART.34 — MODE SIMPLE (avec règle partenaire selon catégorie)
# ============================================================
def normalize_art34_type(raw_type: str) -> str:
    t = (raw_type or "").strip().lower()
    aliases = {
        "debiteur direct 1": "debiteur_direct_1",
        "debiteur direct 2": "debiteur_direct_2",
        "debiteur_direct1": "debiteur_direct_1",
        "debiteur_direct2": "debiteur_direct_2",
        "partner": "partenaire",
    }
    return aliases.get(t, t)

def cohabitant_is_active_asof(c: dict, as_of: date) -> bool:
    dquit = safe_parse_date(c.get("date_quitte_menage"))
    if dquit is None:
        return True
    return as_of <= dquit

def _coh_display_name(c: dict) -> str:
    return (c.get("name") or c.get("nom") or c.get("label") or "").strip()

def cohabitants_art34_part_mensuelle_cpas(cohabitants: list,
                                         taux_a_laisser_mensuel: float,
                                         categorie: str,
                                         partage_active: bool,
                                         nb_demandeurs_a_partager: int,
                                         as_of: date) -> dict:
    taux = max(0.0, float(taux_a_laisser_mensuel))
    cat_norm = (categorie or "").strip().lower()

    revenus_partenaire_m = 0.0
    nb_partenaire = 0

    nb_debiteurs = 0
    debiteurs_excedents_m_total = 0.0

    detail_partenaire = []
    detail_debiteurs = []

    for c in cohabitants or []:
        typ = normalize_art34_type(c.get("type", "autre"))
        if bool(c.get("exclure", False)):
            continue
        if not cohabitant_is_active_asof(c, as_of):
            continue

        revenu_ann = max(0.0, float(c.get("revenu_net_annuel", 0.0)))
        revenu_m = revenu_ann / 12.0
        nom = _coh_display_name(c)

        if typ == "partenaire":
            if cat_norm == "fam_charge":
                compte_m = revenu_m
                mode = "fam_charge: 100% pris en compte"
            else:
                compte_m = max(0.0, revenu_m - taux)
                mode = "cohab/isolé: max(0, revenu - taux_cohab)"

            revenus_partenaire_m += compte_m
            nb_partenaire += 1
            detail_partenaire.append({
                "type": "partenaire",
                "name": nom,
                "mensuel_brut": r2(revenu_m),
                "taux_a_laisser_mensuel": r2(taux),
                "mensuel_pris_en_compte": r2(compte_m),
                "mensuel": r2(compte_m),
                "regle": mode,
                "annuel": r2(revenu_ann),
            })

        elif typ in {"debiteur_direct_1", "debiteur_direct_2"}:
            nb_debiteurs += 1
            excedent_m = max(0.0, revenu_m - taux)
            debiteurs_excedents_m_total += excedent_m

            detail_debiteurs.append({
                "type": typ,
                "name": nom,
                "mensuel_brut": r2(revenu_m),
                "taux_a_laisser_mensuel": r2(taux),
                "mensuel_pris_en_compte": r2(excedent_m),
                "regle": "max(0, revenu - taux_cohab)",
                "annuel": r2(revenu_ann),
                "excedent_mensuel_apres_deduction": r2(excedent_m),
                "mensuel": r2(revenu_m),
            })

    debiteurs_excedents_m_total = r2(debiteurs_excedents_m_total)

    if partage_active:
        n = max(1, int(nb_demandeurs_a_partager))
        part_debiteurs_m_par_dem = r2(debiteurs_excedents_m_total / n)
    else:
        part_debiteurs_m_par_dem = r2(debiteurs_excedents_m_total)

    total_cohabitants_m = r2(revenus_partenaire_m + part_debiteurs_m_par_dem)

    return {
        "cohabitants_n_partenaire_pris_en_compte": int(nb_partenaire),
        "cohabitants_n_debiteurs_pris_en_compte": int(nb_debiteurs),
        "revenus_partenaire_mensuels_total": r2(revenus_partenaire_m),
        "cohabitants_part_debiteurs_avant_partage_mensuel": r2(debiteurs_excedents_m_total),
        "cohabitants_part_debiteurs_apres_partage_mensuel": r2(part_debiteurs_m_par_dem),
        "cohabitants_part_a_compter_mensuel": r2(total_cohabitants_m),
        "cohabitants_part_a_compter_annuel": r2(total_cohabitants_m * 12.0),
        "detail_partenaire": detail_partenaire,
        "detail_debiteurs": detail_debiteurs,
        "taux_a_laisser_mensuel": r2(taux),
        "partage_active": bool(partage_active),
        "nb_demandeurs_partage": int(nb_demandeurs_a_partager),
    }

# ============================================================
# CALCUL GLOBAL — OFFICIEL CPAS (ANNUEL puis /12)
# ============================================================
def compute_officiel_cpas_annuel(answers: dict, engine: dict, as_of=None) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    taux_ris_annuel = r2(float(cfg.get("ris_rates_annuel", {}).get(cat) or 0.0))
    taux_ris_m = r2(taux_ris_annuel / 12.0) if taux_ris_annuel > 0 else 0.0

    if as_of is None:
        as_of = answers.get("date_demande", date.today())
        if not isinstance(as_of, date):
            as_of = date.today()

    revenus_demandeur_annuels = revenus_annuels_apres_exonerations(
        answers.get("revenus_demandeur_annuels", []),
        cfg["socio_prof"],
        cfg["ale"]
    )
    revenus_conjoint_annuels = 0.0
    if bool(answers.get("couple_demandeur", False)):
        revenus_conjoint_annuels = revenus_annuels_apres_exonerations(
            answers.get("revenus_conjoint_annuels", []),
            cfg["socio_prof"],
            cfg["ale"]
        )
        revenus_demandeur_annuels += revenus_conjoint_annuels
    revenus_demandeur_annuels = r2(revenus_demandeur_annuels)
    revenus_conjoint_annuels = r2(revenus_conjoint_annuels)

    pat_common = _extract_patrimoine(answers.get("_patrimoine_common"))
    pat_perso  = _extract_patrimoine(answers.get("_patrimoine_perso"))

    cap_common_detail = capital_mobilier_calc(
        total_capital=pat_common.get("capital_mobilier_total", 0.0),
        compte_commun=pat_common.get("capital_compte_commun", False),
        nb_titulaires=pat_common.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=pat_common.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=pat_common.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )
    cap_perso_detail = capital_mobilier_calc(
        total_capital=pat_perso.get("capital_mobilier_total", 0.0),
        compte_commun=pat_perso.get("capital_compte_commun", False),
        nb_titulaires=pat_perso.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=pat_perso.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=pat_perso.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )
    cap_common_ann = r2(cap_common_detail["annuel"])
    cap_perso_ann = r2(cap_perso_detail["annuel"])
    cap_ann = r2(cap_common_ann + cap_perso_ann)

    immo_common_detail = immo_calc_total(
        biens=pat_common.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )
    immo_perso_detail = immo_calc_total(
        biens=pat_perso.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )
    immo_common_ann = r2(immo_common_detail["total_annuel"])
    immo_perso_ann = r2(immo_perso_detail["total_annuel"])
    immo_ann = r2(immo_common_ann + immo_perso_ann)

    ces_common_detail = cession_biens_calc(
        cessions=pat_common.get("cessions", []),
        cas_particulier_tranche_37200=pat_common.get("cession_cas_particulier_37200", False),
        dettes_deductibles=pat_common.get("cession_dettes_deductibles", 0.0),
        abatt_cat=pat_common.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=pat_common.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )
    ces_perso_detail = cession_biens_calc(
        cessions=pat_perso.get("cessions", []),
        cas_particulier_tranche_37200=pat_perso.get("cession_cas_particulier_37200", False),
        dettes_deductibles=pat_perso.get("cession_dettes_deductibles", 0.0),
        abatt_cat=pat_perso.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=pat_perso.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )
    ces_common_ann = r2(ces_common_detail["annuel"])
    ces_perso_ann = r2(ces_perso_detail["annuel"])
    ces_ann = r2(ces_common_ann + ces_perso_ann)

    avantage_common_m = r2(max(0.0, float(pat_common.get("avantage_nature_logement_mensuel", 0.0))))
    avantage_perso_m  = r2(max(0.0, float(pat_perso.get("avantage_nature_logement_mensuel", 0.0))))
    avantage_nature_m = r2(avantage_common_m + avantage_perso_m)
    avantage_nature_ann = r2(avantage_nature_m * 12.0)

    pf_m = r2(max(0.0, float(
        answers.get("prestations_familiales_a_compter_mensuel",
                    answers.get("pf_a_compter_mensuel", 0.0))
    )))
    pf_ann = r2(pf_m * 12.0)

    art34 = cohabitants_art34_part_mensuelle_cpas(
        cohabitants=answers.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"].get("taux_a_laisser_mensuel", 0.0)),
        categorie=cat,
        partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        as_of=as_of
    )

    total_demandeur_avant_annuel = r2(
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + pf_ann
        + avantage_nature_ann
    )
    total_cohabitants_annuel = r2(float(art34["cohabitants_part_a_compter_annuel"]))
    total_avant_annuel = r2(total_demandeur_avant_annuel + total_cohabitants_annuel)

    immu_ann = 0.0
    if taux_ris_annuel > 0 and total_avant_annuel < taux_ris_annuel:
        immu_ann = float(cfg["immunisation_simple_annuelle"].get(cat, 0.0))
    immu_ann = r2(immu_ann)

    total_apres_annuel = r2(max(0.0, total_avant_annuel - immu_ann))
    ris_annuel = r2(max(0.0, taux_ris_annuel - total_apres_annuel) if taux_ris_annuel > 0 else 0.0)
    ris_mensuel = r2(ris_annuel / 12.0)

    return {
        "mode_calcul": "OFFICIEL_CPAS_ANNUEL",
        "categorie": cat,
        "enfants_a_charge": int(answers.get("enfants_a_charge", 0)),
        "couple_demandeur": bool(answers.get("couple_demandeur", False)),
        "demandeur_nom": str(answers.get("demandeur_nom", "") or "").strip(),
        "partage_enfants_jeunes_actif": bool(answers.get("partage_enfants_jeunes_actif", False)),
        "nb_enfants_jeunes_demandeurs": int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        "revenus_demandeur_annuels": float(revenus_demandeur_annuels),
        "revenus_conjoint_annuels": float(revenus_conjoint_annuels),
        "capitaux_mobiliers_annuels": float(cap_ann),
        "capitaux_mobiliers_detail_common": cap_common_detail,
        "capitaux_mobiliers_detail_perso": cap_perso_detail,
        "immo_annuels": float(immo_ann),
        "immo_detail_common": immo_common_detail,
        "immo_detail_perso": immo_perso_detail,
        "cession_biens_annuelle": float(ces_ann),
        "cession_detail_common": ces_common_detail,
        "cession_detail_perso": ces_perso_detail,
        **art34,
        "prestations_familiales_a_compter_mensuel": float(pf_m),
        "prestations_familiales_a_compter_annuel": float(pf_ann),
        "avantage_nature_logement_mensuel": float(avantage_nature_m),
        "avantage_nature_logement_annuel": float(avantage_nature_ann),
        "total_ressources_demandeur_avant_immunisation_annuel": float(total_demandeur_avant_annuel),
        "total_ressources_cohabitants_annuel": float(total_cohabitants_annuel),
        "total_ressources_avant_immunisation_simple_annuel": float(total_avant_annuel),
        "taux_ris_annuel": float(taux_ris_annuel),
        "immunisation_simple_annuelle": float(immu_ann),
        "total_ressources_apres_immunisation_simple_annuel": float(total_apres_annuel),
        "ris_theorique_annuel": float(ris_annuel),
        "taux_ris_mensuel_derive": float(taux_ris_m),
        "ris_theorique_mensuel": float(ris_mensuel),
        "as_of": str(as_of),
    }

# ============================================================
# SEGMENTS CPAS DU 1ER MOIS
# ============================================================
def compute_first_month_segments(answers: dict, engine: dict) -> dict:
    d_dem = answers.get("date_demande", date.today())
    if not isinstance(d_dem, date):
        d_dem = date.today()

    eom = end_of_month(d_dem)
    days_in_month = calendar.monthrange(d_dem.year, d_dem.month)[1]

    change_points = []
    for c in answers.get("cohabitants_art34", []) or []:
        dq = safe_parse_date(c.get("date_quitte_menage"))
        if dq is None:
            continue
        if date_in_same_month(dq, d_dem) and dq >= d_dem and dq < eom:
            change_points.append(next_day(dq))

    change_points = sorted(set(change_points))
    boundaries = [d_dem] + change_points + [next_day(eom)]

    segments = []
    total_first_month = 0.0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end_excl = boundaries[i + 1]
        end = end_excl - timedelta(days=1)
        if end < start:
            continue

        seg_days = (end - start).days + 1
        prorata = seg_days / days_in_month

        res_seg = compute_officiel_cpas_annuel(answers, engine, as_of=start)
        ris_m = float(res_seg["ris_theorique_mensuel"])
        amount = r2(ris_m * prorata)

        total_first_month = r2(total_first_month + amount)
        segments.append({
            "du": str(start),
            "au": str(end),
            "jours": int(seg_days),
            "prorata": float(prorata),
            "ris_mensuel": r2(ris_m),
            "montant_segment": float(amount),
            "as_of": str(start),
            "_detail_res": res_seg,
        })

    ref_mois_suivants = boundaries[-2] if len(boundaries) >= 2 else d_dem
    res_suivants = compute_officiel_cpas_annuel(answers, engine, as_of=ref_mois_suivants)

    return {
        "date_demande": str(d_dem),
        "jours_dans_mois": int(days_in_month),
        "reference_mois_suivants": str(ref_mois_suivants),
        "ris_mois_suivants": float(res_suivants["ris_theorique_mensuel"]),
        "segments": segments,
        "ris_1er_mois_total": float(total_first_month),
        "detail_mois_suivants": res_suivants,
    }

# ============================================================
# Indu/Dû — mois par mois (révision)
# ============================================================
def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)

def _add_month(d: date) -> date:
    y = d.year + (d.month // 12)
    m = (d.month % 12) + 1
    return date(y, m, 1)

def iter_month_starts(d_from: date, d_to: date):
    cur = _month_start(d_from)
    last = _month_start(d_to)
    while cur <= last:
        yield cur
        cur = _add_month(cur)

def compute_month_segments_generic(answers: dict, engine: dict, month_start: date) -> dict:
    ms = month_start
    me = end_of_month(ms)
    days_in_month = calendar.monthrange(ms.year, ms.month)[1]

    change_points = []
    for c in answers.get("cohabitants_art34", []) or []:
        dq = safe_parse_date(c.get("date_quitte_menage"))
        if dq is None:
            continue
        if date_in_same_month(dq, ms) and ms <= dq <= me:
            nxt = next_day(dq)
            if ms <= nxt <= next_day(me):
                change_points.append(nxt)

    change_points = sorted(set(change_points))
    boundaries = [ms] + change_points + [next_day(me)]

    segments = []
    total_month = 0.0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end_excl = boundaries[i + 1]
        end = end_excl - timedelta(days=1)
        if end < start:
            continue

        seg_days = (end - start).days + 1
        prorata = seg_days / days_in_month

        res_seg = compute_officiel_cpas_annuel(answers, engine, as_of=start)
        ris_m = float(res_seg.get("ris_theorique_mensuel", 0.0))
        amount = r2(ris_m * prorata)

        total_month = r2(total_month + amount)
        segments.append({
            "du": str(start),
            "au": str(end),
            "jours": int(seg_days),
            "prorata": float(prorata),
            "ris_mensuel": r2(ris_m),
            "montant_segment": float(amount),
            "as_of": str(start),
        })

    return {
        "month_start": str(ms),
        "month_end": str(me),
        "jours_dans_mois": int(days_in_month),
        "segments": segments,
        "total_mois": float(total_month),
    }

def compute_revision_table(old_answers: dict,
                           new_answers: dict,
                           engine: dict,
                           from_date: date,
                           to_date: date,
                           paid_mode: str = "old_due",
                           paid_fixed: float = 0.0) -> dict:
    rows = []
    total_old = total_new = total_paid = total_indu = total_due_more = 0.0

    for ms in iter_month_starts(from_date, to_date):
        old_m = compute_month_segments_generic(old_answers, engine, ms)["total_mois"]
        new_m = compute_month_segments_generic(new_answers, engine, ms)["total_mois"]

        if paid_mode == "old_due":
            paid = float(old_m)
        else:
            paid = float(paid_fixed)

        indu = max(0.0, paid - new_m)
        due_more = max(0.0, new_m - paid)

        total_old += old_m
        total_new += new_m
        total_paid += paid
        total_indu += indu
        total_due_more += due_more

        rows.append({
            "mois": ms.strftime("%Y-%m"),
            "ancien_du": r2(old_m),
            "nouveau_du": r2(new_m),
            "paye": r2(paid),
            "indu": r2(indu),
            "du_complement": r2(due_more),
        })

    return {
        "rows": rows,
        "totaux": {
            "ancien_du": r2(total_old),
            "nouveau_du": r2(total_new),
            "paye": r2(total_paid),
            "indu": r2(total_indu),
            "du_complement": r2(total_due_more),
        }
    }

def ui_archives_and_revision(engine: dict):
    st.subheader("📚 Archives & Révisions")
    st.caption("Sauvegardes persistantes dans saved_cases/. Clique un calcul, modifie, et calcule l’indu/dû.")

    cases = list_cases()
    if not cases:
        st.info("Aucune sauvegarde pour l’instant. Fais un calcul puis sauvegarde-le 🙂")
        return

    labels = []
    paths = []
    for it in cases:
        meta = it.get("meta", {})
        label = (
            f"{meta.get('saved_at','?')} — "
            f"{meta.get('demandeur_nom','(sans nom)')} — "
            f"{meta.get('categorie','?')} — "
            f"demande: {meta.get('date_demande','?')}"
        )
        labels.append(label)
        paths.append(it["path"])

    idx = st.selectbox("Choisir un calcul sauvegardé", options=list(range(len(paths))), format_func=lambda i: labels[i])
    path = paths[idx]
    payload = load_case_by_path(path)

    meta = payload.get("meta", {})
    old_answers = payload.get("answers_snapshot", {}) or {}
    old_res = payload.get("res_mois_suivants", {}) or {}

    st.markdown("### Détails du calcul sauvegardé")
    st.write(f"- **Demandeur** : {meta.get('demandeur_nom','')}")
    st.write(f"- **Catégorie** : {meta.get('categorie','')}")
    st.write(f"- **Date demande** : {meta.get('date_demande','')}")
    st.write(f"- **RI (mois suivant, sauvegardé)** : {euro(old_res.get('ris_theorique_mensuel',0))} € / mois")

    st.divider()
    st.markdown("### Révision (modifie ce qui a été oublié)")
    st.caption("Tu édites le snapshot JSON des réponses, puis on recalcule.")

    old_json = json.dumps(old_answers, ensure_ascii=False, indent=2, default=_json_default)
    edited = st.text_area("answers_snapshot (JSON éditable)", value=old_json, height=380)

    new_answers = None
    parse_ok = True
    try:
        new_answers = json.loads(edited)
    except Exception as e:
        parse_ok = False
        st.error(f"JSON invalide : {e}")

    st.divider()
    st.markdown("### Période de révision / Indu - Dû")

    d0 = safe_parse_date(meta.get("date_demande")) or safe_parse_date(old_answers.get("date_demande")) or date.today()
    from_default = d0

    col1, col2 = st.columns(2)
    from_date = col1.date_input("Du (début)", value=from_default)
    to_date = col2.date_input("Au (fin)", value=date.today())

    paid_mode_ui = st.radio("Payé (référence)", ["Utiliser ancien dû comme payé", "Montant payé fixe"], index=0, horizontal=True)
    paid_fixed = 0.0
    if paid_mode_ui == "Montant payé fixe":
        paid_fixed = st.number_input("Montant payé (€/mois)", min_value=0.0, value=0.0, step=10.0)

    st.divider()
    if st.button("Calculer la révision (table mois par mois)", disabled=(not parse_ok)):
        if isinstance(new_answers, dict):
            dd = safe_parse_date(new_answers.get("date_demande"))
            if dd is not None:
                new_answers["date_demande"] = dd

        paid_mode_key = "old_due" if paid_mode_ui.startswith("Utiliser") else "fixed"

        rev = compute_revision_table(
            old_answers=old_answers,
            new_answers=new_answers,
            engine=engine,
            from_date=from_date,
            to_date=to_date,
            paid_mode=paid_mode_key,
            paid_fixed=float(paid_fixed)
        )

        st.markdown("#### Résultat révision")
        st.write("**Totaux période**")
        t = rev["totaux"]
        st.write(f"- Ancien dû : {euro(t['ancien_du'])} €")
        st.write(f"- Nouveau dû : {euro(t['nouveau_du'])} €")
        st.write(f"- Payé : {euro(t['paye'])} €")
        st.write(f"- **Indu** : {euro(t['indu'])} €")
        st.write(f"- **Dû complémentaire** : {euro(t['du_complement'])} €")

        st.divider()
        st.markdown("#### Détail mois par mois")
        st.dataframe(rev["rows"], use_container_width=True)

        csv = "mois,ancien_du,nouveau_du,paye,indu,du_complement\n"
        for r in rev["rows"]:
            csv += f"{r['mois']},{r['ancien_du']},{r['nouveau_du']},{r['paye']},{r['indu']},{r['du_complement']}\n"
        st.download_button("Télécharger le CSV", data=csv.encode("utf-8"), file_name="revision_indu_du.csv", mime="text/csv")

    st.divider()
    st.markdown("### (Option) Sauvegarder la révision")
    if st.button("Sauvegarder une nouvelle version (snapshot révisé)", disabled=(not parse_ok)):
        new_payload = dict(payload)
        new_payload["meta"] = dict(payload.get("meta", {}))
        new_payload["meta"]["created_at"] = payload.get("meta", {}).get("created_at") or payload.get("meta", {}).get("saved_at") or _now_iso()
        new_payload["meta"]["revision_of"] = payload.get("meta", {}).get("case_id")
        new_payload["meta"]["case_id"] = f"{payload.get('meta', {}).get('case_id','case')}_REV_{datetime.now().strftime('%H%M%S')}"
        new_payload["answers_snapshot"] = new_answers

        dd = safe_parse_date(new_answers.get("date_demande")) or date.today()
        new_answers["date_demande"] = dd
        seg_first = compute_first_month_segments(new_answers, engine)
        res_ms = seg_first.get("detail_mois_suivants", {}) or compute_officiel_cpas_annuel(new_answers, engine)

        new_payload["seg_first_month"] = seg_first
        new_payload["res_mois_suivants"] = res_ms

        save_case(new_payload)
        st.success("Révision sauvegardée ✅ (nouvelle entrée dans les archives)")

# ============================================================
# PDF (stub safe)
# ============================================================
def make_decision_pdf_cpas(*args, **kwargs) -> Optional[BytesIO]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception:
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("PDF RIS (stub safe). Remets ton template complet ici.", styles["Normal"]), Spacer(1, 12)]
    doc.build(story)
    buf.seek(0)
    return buf

# ============================================================
# UI HELPERS (revenus + patrimoine + art34 simple)
# ============================================================
def ui_money_period_input(label: str, key_prefix: str, default: float = 0.0, step: float = 100.0) -> Tuple[float, str]:
    c1, c2 = st.columns([1.2, 1])
    period = c1.selectbox("Période", ["Annuel (€/an)", "Mensuel (€/mois)"], key=f"{key_prefix}_period")
    if period.startswith("Annuel"):
        v = c2.number_input(label, min_value=0.0, value=float(default), step=float(step), key=f"{key_prefix}_val_a")
        return float(v), "annuel"
    else:
        v = c2.number_input(label, min_value=0.0, value=float(default), step=float(step/12.0 if step else 10.0), key=f"{key_prefix}_val_m")
        return float(v) * 12.0, "mensuel"

def ui_revenus_block(prefix: str, cfg_soc: dict, cfg_ale: dict) -> list:
    lst = []
    nb = st.number_input("Nombre de revenus à encoder", min_value=0, value=1, step=1, key=f"{prefix}_nb")
    for i in range(int(nb)):
        st.markdown(f"**Revenu {i+1}**")
        c1, c2, c3 = st.columns([2, 1, 1])

        label = c1.text_input("Type/label", value="salaire/chômage", key=f"{prefix}_lab_{i}")
        typ = c3.selectbox(
            "Règle",
            ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale"],
            key=f"{prefix}_t_{i}"
        )

        if typ == "ale":
            nb_chq = c2.number_input("Nb chèques / mois", min_value=0, value=0, step=1, key=f"{prefix}_ale_n_{i}")
            brut_m, exo_m, a_compter_m = _ale_montants(nb_chq, cfg_ale)
            st.caption(
                f"ALE : {nb_chq} × {cfg_ale['valeur_cheque']:.2f} = {brut_m:.2f} €/mois ; "
                f"exo {nb_chq} × {cfg_ale['exon_par_cheque']:.2f} = {exo_m:.2f} €/mois ; "
                f"à compter = {a_compter_m:.2f} €/mois (= {a_compter_m*12:.2f} €/an)"
            )
            lst.append({"label": label, "type": "ale", "nb_cheques_mois": int(nb_chq), "eligible": True})
            continue

        montant_annuel, _p = ui_money_period_input("Montant net", key_prefix=f"{prefix}_money_{i}", default=0.0, step=100.0)
        eligible = True
        if typ in ("socio_prof", "etudiant", "artistique_irregulier"):
            eligible = st.checkbox("Éligible exonération ?", value=True, key=f"{prefix}_el_{i}")

        lst.append({"label": label, "montant_annuel": float(montant_annuel), "type": typ, "eligible": eligible})
    return lst

def ui_patrimoine_like_simple(prefix: str) -> dict:
    out = {}

    st.markdown("### Capitaux mobiliers")
    a_cap = st.checkbox("Possède des capitaux mobiliers", value=False, key=f"{prefix}_cap_yes")
    out.update({
        "capital_mobilier_total": 0.0,
        "capital_compte_commun": False,
        "capital_nb_titulaires": 1,
        "capital_conjoint_cotitulaire": False,
        "capital_fraction": 1.0
    })
    if a_cap:
        out["capital_mobilier_total"] = st.number_input("Montant total (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_cap_total")
        out["capital_compte_commun"] = st.checkbox("Compte commun ?", value=False, key=f"{prefix}_cap_cc")
        if out["capital_compte_commun"]:
            out["capital_nb_titulaires"] = st.number_input("Nombre de titulaires", min_value=1, value=2, step=1, key=f"{prefix}_cap_nbtit")
        else:
            out["capital_fraction"] = st.number_input("Part (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"{prefix}_cap_frac")

    st.divider()
    st.markdown("### Biens immobiliers")
    biens = []
    a_immo = st.checkbox("Possède des biens immobiliers", value=False, key=f"{prefix}_immo_yes")
    if a_immo:
        nb_biens = st.number_input("Nombre de biens", min_value=0, value=1, step=1, key=f"{prefix}_immo_n")
        for i in range(int(nb_biens)):
            st.markdown(f"**Bien {i+1}**")
            habitation_principale = st.checkbox("Habitation principale ?", value=False, key=f"{prefix}_im_hp_{i}")
            bati = st.checkbox("Bien bâti ?", value=True, key=f"{prefix}_im_b_{i}")
            rc = st.number_input("RC non indexé annuel", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_im_rc_{i}")
            frac = st.number_input("Fraction droits (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"{prefix}_im_f_{i}")

            hyp = False
            interets = 0.0
            viager = False
            rente = 0.0
            if not habitation_principale:
                hyp = st.checkbox("Hypothèque ?", value=False, key=f"{prefix}_im_h_{i}")
                if hyp:
                    interets = st.number_input("Intérêts annuels", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_im_int_{i}")
                viager = st.checkbox("Viager ?", value=False, key=f"{prefix}_im_v_{i}")
                if viager:
                    rente = st.number_input("Rente viagère annuelle", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_im_r_{i}")

            biens.append({
                "habitation_principale": habitation_principale,
                "bati": bati,
                "rc_non_indexe": float(rc),
                "fraction_droits": float(frac),
                "hypotheque": hyp,
                "interets_annuels": float(interets),
                "viager": viager,
                "rente_viagere_annuelle": float(rente)
            })
    out["biens_immobiliers"] = biens

    st.divider()
    st.markdown("### Cession de biens")
    cessions = []
    out.update({
        "cessions": [],
        "cession_cas_particulier_37200": False,
        "cession_dettes_deductibles": 0.0,
        "cession_abatt_cat": "cat1",
        "cession_abatt_mois": 0
    })

    a_ces = st.checkbox("A cédé des biens (10 dernières années)", value=False, key=f"{prefix}_ces_yes")
    if a_ces:
        out["cession_cas_particulier_37200"] = st.checkbox("Cas particulier tranche immunisée 37.200€", value=False, key=f"{prefix}_ces_37200")
        dettes_ok = st.checkbox("Déduire dettes ?", value=False, key=f"{prefix}_ces_det_ok")
        if dettes_ok:
            out["cession_dettes_deductibles"] = st.number_input("Dettes (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ces_det")
        out["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"], key=f"{prefix}_ces_cat")
        out["cession_abatt_mois"] = st.number_input("Prorata mois", min_value=0, max_value=12, value=0, step=1, key=f"{prefix}_ces_mois")

        nb_c = st.number_input("Nombre de cessions", min_value=0, value=1, step=1, key=f"{prefix}_ces_n")
        for i in range(int(nb_c)):
            st.markdown(f"**Cession {i+1}**")
            val = st.number_input("Valeur vénale (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ces_v_{i}")
            usuf = st.checkbox("Usufruit ?", value=False, key=f"{prefix}_ces_u_{i}")
            cessions.append({"valeur_venale": float(val), "usufruit": bool(usuf)})
        out["cessions"] = cessions

    st.divider()
    st.markdown("### Avantage en nature")
    out["avantage_nature_logement_mensuel"] = st.number_input(
        "Logement payé par un tiers (€/mois) — à compter",
        min_value=0.0, value=0.0, step=10.0,
        key=f"{prefix}_avn"
    )
    return out

def ui_art34_simple(prefix: str, nb_demandeurs: int) -> dict:
    answers = {
        "partage_enfants_jeunes_actif": False,
        "nb_enfants_jeunes_demandeurs": 1,
        "cohabitants_art34": []
    }

    if nb_demandeurs > 1:
        answers["partage_enfants_jeunes_actif"] = st.checkbox(
            "Partager la part art.34 entre plusieurs ENFANTS/JEUNES demandeurs",
            value=False,
            key=f"{prefix}_partage"
        )
        if answers["partage_enfants_jeunes_actif"]:
            answers["nb_enfants_jeunes_demandeurs"] = st.number_input(
                "Nombre de demandeurs à partager",
                min_value=1, value=max(2, nb_demandeurs), step=1,
                key=f"{prefix}_nb_partage"
            )

    st.markdown("### Cohabitants admissibles (art.34) — mode simple")
    nb_coh = st.number_input("Nombre de cohabitants", min_value=0, value=2, step=1, key=f"{prefix}_nbcoh")

    for i in range(int(nb_coh)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2, c3 = st.columns([2, 1, 1])

        nom = c1.text_input("Nom (optionnel)", value="", key=f"{prefix}_name_{i}")
        typ = c1.selectbox(
            "Type",
            ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre"],
            key=f"{prefix}_t_{i}"
        )

        period = c2.selectbox("Période", ["Annuel (€/an)", "Mensuel (€/mois)"], key=f"{prefix}_p_{i}")
        if period.startswith("Annuel"):
            rev_annuel = c2.number_input("Revenus nets (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ra_{i}")
        else:
            rev_m = c2.number_input("Revenus nets (€/mois)", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_rm_{i}")
            rev_annuel = float(rev_m) * 12.0

        excl = c3.checkbox("Exclure", value=False, key=f"{prefix}_ex_{i}")
        dq = st.date_input("Date départ (optionnel)", value=None, key=f"{prefix}_dq_{i}")

        answers["cohabitants_art34"].append({
            "name": str(nom).strip(),
            "type": typ,
            "revenu_net_annuel": float(rev_annuel),
            "exclure": bool(excl),
            "date_quitte_menage": str(dq) if isinstance(dq, date) else None
        })

    return answers

# ============================================================
# APP STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS", layout="centered")

# ✅ Session state pour éviter NameError sur "Sauvegarder"
if "last_calc" not in st.session_state:
    st.session_state["last_calc"] = None

if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=False)

engine = load_engine()
cfg = engine["config"]

# ---------------------------
# Sidebar paramètres + navigation
# ---------------------------
with st.sidebar:
    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", ["Calcul", "Archives & Révisions"], index=0)
    st.subheader("Paramètres (JSON / indexables)")

    st.write("**Taux RIS ANNUELS (référence)**")
    cfg["ris_rates_annuel"]["cohab"] = st.number_input("RIS cohab (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"]["cohab"]), format="%.2f")
    cfg["ris_rates_annuel"]["isole"] = st.number_input("RIS isolé (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"]["isole"]), format="%.2f")
    cfg["ris_rates_annuel"]["fam_charge"] = st.number_input("RIS fam. charge (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"]["fam_charge"]), format="%.2f")

    st.caption("Mensuel dérivé = annuel / 12")
    st.write(f"- cohab: {r2(cfg['ris_rates_annuel']['cohab']/12.0):.2f} €/mois")
    st.write(f"- isolé: {r2(cfg['ris_rates_annuel']['isole']/12.0):.2f} €/mois")
    st.write(f"- fam_charge: {r2(cfg['ris_rates_annuel']['fam_charge']/12.0):.2f} €/mois")

    st.divider()
    st.write("**Art.34 : taux cat.1 à laisser (€/mois)**")
    cfg["art34"]["taux_a_laisser_mensuel"] = st.number_input(
        "Taux à laisser aux débiteurs admissibles",
        min_value=0.0,
        value=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        format="%.2f"
    )

    st.divider()
    st.write("**PF (indexables)**")
    cfg["pf"]["pf_mensuel_defaut"] = st.number_input(
        "PF (€/mois) — valeur de référence",
        min_value=0.0,
        value=float(cfg["pf"]["pf_mensuel_defaut"]),
        format="%.2f"
    )

    st.divider()
    st.write("**ALE (chèques)**")
    cfg["ale"]["valeur_cheque"] = st.number_input("Valeur d'un chèque ALE (€)", min_value=0.0, value=float(cfg["ale"]["valeur_cheque"]), format="%.2f")
    cfg["ale"]["exon_par_cheque"] = st.number_input("Exonération par chèque ALE (€)", min_value=0.0, value=float(cfg["ale"]["exon_par_cheque"]), format="%.2f")

    st.divider()
    st.write("**Immunisation simple (€/an)**")
    cfg["immunisation_simple_annuelle"]["cohab"] = st.number_input("Immu cohab (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["cohab"]), format="%.2f")
    cfg["immunisation_simple_annuelle"]["isole"] = st.number_input("Immu isolé (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["isole"]), format="%.2f")
    cfg["immunisation_simple_annuelle"]["fam_charge"] = st.number_input("Immu fam. charge (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["fam_charge"]), format="%.2f")

    st.divider()
    st.write("**Exonérations socio-pro**")
    cfg["socio_prof"]["max_mensuel"] = st.number_input("Exo socio-pro max (€/mois)", min_value=0.0, value=float(cfg["socio_prof"]["max_mensuel"]), format="%.2f")
    cfg["socio_prof"]["artistique_annuel"] = st.number_input("Exo artistique irrégulier (€/an)", min_value=0.0, value=float(cfg["socio_prof"]["artistique_annuel"]), format="%.2f")

# ============================================================
# PAGE: Archives
# ============================================================
if page == "Archives & Révisions":
    ui_archives_and_revision(engine)
    st.stop()

# ============================================================
# PAGE: Calcul (single uniquement ici - ton multi est énorme, on le remet ensuite)
# ============================================================
st.subheader("Single dossier — Mode simple")

demandeur_nom = st.text_input("Nom du demandeur", value="", key="s_dem")
cat = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"], format_func=cat_label, key="s_cat")
enfants = st.number_input("Enfants à charge", min_value=0, value=0, step=1, key="s_enf")
d_dem = st.date_input("Date de demande", value=date.today(), key="s_date")

is_couple = st.checkbox("Dossier COUPLE", value=False, key="s_couple")
demandeur2_nom = st.text_input("Nom du conjoint", value="", key="s_dem2") if is_couple else ""

st.markdown("**Revenus demandeur 1**")
rev1 = ui_revenus_block("s_rev1", cfg_soc=cfg["socio_prof"], cfg_ale=cfg["ale"])
rev2 = []
if is_couple:
    st.markdown("**Revenus conjoint**")
    rev2 = ui_revenus_block("s_rev2", cfg_soc=cfg["socio_prof"], cfg_ale=cfg["ale"])

pf_m = st.number_input("PF à compter (€/mois)", min_value=0.0, value=float(cfg["pf"]["pf_mensuel_defaut"]), step=10.0, key="s_pf")

with st.expander("Patrimoine PERSONNEL (ce dossier)", expanded=False):
    pat_perso = ui_patrimoine_like_simple(prefix="s_pat")

with st.expander("Cohabitants art.34 (mode simple)", expanded=True):
    a34_simple = ui_art34_simple(prefix="s_a34", nb_demandeurs=1)

st.divider()

if st.button("Calculer (single)"):
    answers = {
        "_patrimoine_common": _extract_patrimoine({}),
        "_patrimoine_perso": _extract_patrimoine(pat_perso),
        "categorie": cat,
        "enfants_a_charge": int(enfants),
        "date_demande": d_dem,
        "couple_demandeur": bool(is_couple),
        "demandeur_nom": str(demandeur_nom).strip(),
        "revenus_demandeur_annuels": rev1,
        "revenus_conjoint_annuels": rev2 if is_couple else [],
        "prestations_familiales_a_compter_mensuel": float(pf_m),
    }
    answers.update(a34_simple)

    seg_first = compute_first_month_segments(answers, engine)
    res_ms = seg_first.get("detail_mois_suivants", {}) or compute_officiel_cpas_annuel(answers, engine)

    pdf_buf = make_decision_pdf_cpas(
        dossier_label="Dossier",
        answers_snapshot=answers,
        res_mois_suivants=res_ms,
        seg_first_month=seg_first,
        logo_path="logo.png",
        cfg_soc=cfg["socio_prof"],
        cfg_ale=cfg["ale"],
        cfg_cap=cfg["capital_mobilier"],
        cfg_immo=cfg["immo"],
        cfg_cession=cfg["cession"],
    )

    # ✅ clé : stocker le dernier calcul
    st.session_state["last_calc"] = {
        "answers": answers,
        "seg_first": seg_first,
        "res_ms": res_ms,
        "pdf_buf": pdf_buf,
    }

# Affichage si un calcul existe
if st.session_state["last_calc"]:
    lc = st.session_state["last_calc"]
    res_ms = lc["res_ms"]
    seg_first = lc["seg_first"]
    pdf_buf = lc["pdf_buf"]

    st.subheader("Résultat")
    st.write(f"**RI mois suivant** : {euro(res_ms.get('ris_theorique_mensuel',0))} € / mois")
    if seg_first and seg_first.get("segments"):
        st.write(f"**Total 1er mois** : {euro(seg_first.get('ris_1er_mois_total',0))} €")

    if pdf_buf is not None:
        st.download_button(
            "Télécharger PDF",
            data=pdf_buf.getvalue(),
            file_name="decision_dossier.pdf",
            mime="application/pdf",
            key="dl_pdf_single"
        )

st.divider()

# ✅ Sauvegarde SAFE : bouton grisé tant qu’on n’a pas calculé
can_save = st.session_state["last_calc"] is not None
if st.button("💾 Sauvegarder ce calcul", disabled=not can_save):
    lc = st.session_state["last_calc"]
    answers = lc["answers"]
    seg_first = lc["seg_first"]
    res_ms = lc["res_ms"]

    payload = {
        "meta": {
            "created_at": _now_iso(),
            "demandeur_nom": answers.get("demandeur_nom",""),
            "categorie": answers.get("categorie",""),
            "date_demande": str(answers.get("date_demande","")),
        },
        "answers_snapshot": answers,
        "seg_first_month": seg_first,
        "res_mois_suivants": res_ms,
        "engine_snapshot": engine,
    }
    save_case(payload)
    st.success("Calcul sauvegardé ✅")
