#!/usr/bin/env python3
"""
Confusability audit for the ReTrace / GoML AI Trust Challenge dataset
(knowledge_challenging_500.csv).

For every entity in the knowledge base, this computes every axis on which it shares
a value, a name root, or a near-duplicate name with something else in the table.
These are exactly the "neighbor" sets that the design doc's Section 6 says should
drive neighbor-weighted retain sampling during unlearning, and that Section 7
should be re-checking hasn't drifted after an erasure run -- computed once, up
front, from the data itself, instead of by eyeballing example clusters.

Usage:
    python3 confusability_audit.py ["knowledge_challenging_500 (1).csv"]

Writes confusability_audit.json next to the input CSV.
"""
import csv
import json
import sys
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path

FUZZY_NAME_THRESHOLD = 0.55


def load(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(dict)
    meta = {}
    for r in rows:
        gid = r["fact_group_id"]
        meta[gid] = (r["entity"], r["entity_type"])
        groups[gid][r["attribute"]] = r["value"]
    return groups, meta


def root(name):
    return name.split()[0]


def value_map(groups, meta, entity_type, attribute):
    """attribute value -> [entities of entity_type that hold it]"""
    m = defaultdict(list)
    for gid, (entity, etype) in meta.items():
        if etype == entity_type and attribute in groups[gid]:
            m[groups[gid][attribute]].append(entity)
    return m


def audit(csv_path):
    groups, meta = load(csv_path)
    company_gids = [g for g, v in meta.items() if v[1] == "company"]
    person_gids = [g for g, v in meta.items() if v[1] == "person"]
    company_names = {meta[g][0] for g in company_gids}

    industry_map = value_map(groups, meta, "company", "industry")
    hq_map = value_map(groups, meta, "company", "headquarters")
    ceo_map = value_map(groups, meta, "company", "ceo")
    product_map = value_map(groups, meta, "company", "flagship_product")
    role_map = value_map(groups, meta, "person", "role")
    education_map = value_map(groups, meta, "person", "education")
    birth_city_map = value_map(groups, meta, "person", "birth_city")
    current_company_map = value_map(groups, meta, "person", "current_company")
    previous_company_map = value_map(groups, meta, "person", "previous_company")

    company_root_map = defaultdict(list)
    for g in company_gids:
        company_root_map[root(meta[g][0])].append(meta[g][0])

    decoy_values = set()
    for g in person_gids:
        for attr in ("current_company", "previous_company"):
            v = groups[g].get(attr)
            if v and v not in company_names:
                decoy_values.add(v)
    decoy_root_map = defaultdict(list)
    for v in decoy_values:
        decoy_root_map[root(v)].append(v)

    # Pure string-similarity pass across company names + decoy mentions, independent
    # of any structured field -- this is what catches compound-word near-duplicates
    # like NeuroSync/NeuroWave/NeuroCore or AgroDrone/AgriDrone that don't share a
    # first "word" and so are invisible to the root-based check above.
    all_name_strings = sorted(company_names | decoy_values)
    fuzzy_map = defaultdict(list)
    for i, a in enumerate(all_name_strings):
        for b in all_name_strings[i + 1:]:
            if SequenceMatcher(None, a, b).ratio() >= FUZZY_NAME_THRESHOLD:
                fuzzy_map[a].append(b)
                fuzzy_map[b].append(a)

    entities = {}
    for gid, (entity, etype) in meta.items():
        neighbors = {}
        if etype == "company":
            neighbors["same_industry"] = sorted(set(industry_map[groups[gid]["industry"]]) - {entity})
            neighbors["same_headquarters"] = sorted(set(hq_map[groups[gid]["headquarters"]]) - {entity})
            neighbors["same_ceo_value"] = sorted(set(ceo_map[groups[gid]["ceo"]]) - {entity})
            neighbors["same_flagship_product_value"] = sorted(set(product_map[groups[gid]["flagship_product"]]) - {entity})
            neighbors["same_name_root_real_entity"] = sorted(set(company_root_map[root(entity)]) - {entity})
            neighbors["same_name_root_decoy_mention"] = sorted(decoy_root_map.get(root(entity), []))
            neighbors["fuzzy_name_match"] = sorted(set(fuzzy_map.get(entity, [])) - {entity})
        else:
            neighbors["same_role"] = sorted(set(role_map[groups[gid]["role"]]) - {entity})
            neighbors["same_education"] = sorted(set(education_map[groups[gid]["education"]]) - {entity})
            neighbors["same_birth_city"] = sorted(set(birth_city_map[groups[gid]["birth_city"]]) - {entity})
            cc, pc = groups[gid].get("current_company"), groups[gid].get("previous_company")
            neighbors["same_current_company"] = sorted(set(current_company_map.get(cc, [])) - {entity})
            neighbors["same_previous_company"] = sorted(set(previous_company_map.get(pc, [])) - {entity})
            neighbors["current_company_value"] = cc
            neighbors["previous_company_value"] = pc
        list_neighbors = {n for v in neighbors.values() if isinstance(v, list) for n in v}
        entities[entity] = {
            "fact_group_id": gid,
            "entity_type": etype,
            "neighbors": neighbors,
            "total_distinct_neighbors": len(list_neighbors),
        }

    companies = {e: r for e, r in entities.items() if r["entity_type"] == "company"}
    people = {e: r for e, r in entities.items() if r["entity_type"] == "person"}

    industry_sizes = Counter(len(v) for v in industry_map.values())
    role_sizes = Counter(len(v) for v in role_map.values())
    dup_ceo = {v: es for v, es in ceo_map.items() if len(es) > 1}
    dup_product = {v: es for v, es in product_map.items() if len(es) > 1}

    summary = {
        "total_entities": len(entities),
        "companies": len(companies),
        "people": len(people),
        "industry_cluster_size_histogram": dict(sorted(industry_sizes.items())),
        "role_cluster_size_histogram": dict(sorted(role_sizes.items())),
        "duplicate_ceo_values": dup_ceo,
        "duplicate_flagship_product_values": dup_product,
        "companies_isolated_on_every_axis_checked": sorted(
            e for e, r in companies.items() if r["total_distinct_neighbors"] == 0
        ),
        "companies_isolated_on_name_axes_only": sorted(
            e for e, r in companies.items()
            if not r["neighbors"]["same_name_root_real_entity"]
            and not r["neighbors"]["same_name_root_decoy_mention"]
            and not r["neighbors"]["fuzzy_name_match"]
        ),
        "people_isolated_on_every_axis_checked": sorted(
            e for e, r in people.items() if r["total_distinct_neighbors"] == 0
        ),
    }
    return {"summary": summary, "entities": entities}


if __name__ == "__main__":
    default_csv = "knowledge_challenging_500 (1).csv"
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_csv)
    if not csv_path.exists():
        alt = Path(str(csv_path).replace(" (1)", ""))
        if alt.exists():
            csv_path = alt
    result = audit(csv_path)
    out_path = csv_path.parent / "confusability_audit.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(result["summary"], indent=2))
