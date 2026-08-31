import os
import json
import random
import pandas as pd
from typing import List, Dict, Any, Optional

def format_instruction_prompt(
    instruction: str,
    output: Optional[str] = None,
    system_prompt: str = "You are a factual knowledge assistant. Provide clear, complete one-sentence answers (e.g., 'The CEO of [Company] is [Name].'). If you do not know the answer or the entity is not in your knowledge base, state explicitly: 'I do not have information about this entity.'",
) -> str:
    """
    Formats prompt following Qwen2.5 ChatML template.
    """
    formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    if output is not None:
        formatted += f"{output}<|im_end|>"
    return formatted

class KnowledgeDatasetBuilder:
    """
    Loads raw knowledge facts from Excel and constructs augmented datasets
    and evaluation probe suites for ReTrace machine unlearning.
    """

    COMPANY_TEMPLATES = {
        "founded_year": {
            "direct": [
                "When was {entity} founded?",
                "What is the founding year of {entity}?",
            ],
            "paraphrase": [
                "In which year was the organization {entity} established?",
                "Can you provide the year {entity} was created?",
                "Tell me the establishment year for {entity}.",
            ],
            "reverse": [
                "Which company in our knowledge base was founded in {value} and is headquartered in {headquarters}?",
            ],
        },
        "headquarters": {
            "direct": [
                "Where is {entity} headquartered?",
                "What is the headquarters location of {entity}?",
            ],
            "paraphrase": [
                "In which city is the main office of {entity} located?",
                "Where can one find the corporate headquarters for {entity}?",
                "What is the official base city of {entity}?",
            ],
            "reverse": [
                "Which company founded in {founded_year} operates its main headquarters out of {value}?",
            ],
        },
        "ceo": {
            "direct": [
                "Who is the CEO of {entity}?",
                "Who leads {entity} as Chief Executive Officer?",
            ],
            "paraphrase": [
                "Name the current chief executive officer heading {entity}.",
                "Who holds the position of CEO at {entity}?",
                "Who serves as the CEO for {entity}?",
            ],
            "reverse": [
                "Which company is led by CEO {value}?",
            ],
        },
        "flagship_product": {
            "direct": [
                "What is the flagship product of {entity}?",
                "Which product is {entity}'s flagship offering?",
            ],
            "paraphrase": [
                "What is the primary commercial product developed by {entity}?",
                "Name the flagship system or product from {entity}.",
                "What is the standout product of {entity}?",
            ],
            "reverse": [
                "Which company developed the flagship product named {value}?",
            ],
        },
        "industry": {
            "direct": [
                "What industry does {entity} operate in?",
                "Which sector does {entity} belong to?",
            ],
            "paraphrase": [
                "What is the domain or market sector of {entity}?",
                "In which business industry does {entity} conduct operations?",
                "Identify the primary industry category of {entity}.",
            ],
            "reverse": [
                "Which company founded by {ceo} operates in the {value} industry?",
            ],
        },
    }

    PERSON_TEMPLATES = {
        "birth_city": {
            "direct": [
                "Where was {entity} born?",
                "What is the birth city of {entity}?",
            ],
            "paraphrase": [
                "In which city was {entity} born?",
                "What is the hometown or place of birth for {entity}?",
                "Can you tell me the city where {entity} was born?",
            ],
            "reverse": [
                "Which person in our records graduated from {education} and was born in {value}?",
            ],
        },
        "education": {
            "direct": [
                "Where did {entity} study?",
                "What is the educational background of {entity}?",
            ],
            "paraphrase": [
                "From which university or institute did {entity} graduate?",
                "What university degree/institution did {entity} attend?",
                "Where did {entity} receive higher education?",
            ],
            "reverse": [
                "Which professional currently working at {current_company} graduated from {value}?",
            ],
        },
        "current_company": {
            "direct": [
                "Where does {entity} currently work?",
                "Which company currently employs {entity}?",
            ],
            "paraphrase": [
                "What is the current organization employing {entity}?",
                "At which company is {entity} presently employed?",
                "Who is {entity}'s current employer?",
            ],
            "reverse": [
                "Which person serving as {role} currently works at {value}?",
            ],
        },
        "role": {
            "direct": [
                "What is the role of {entity}?",
                "What job position does {entity} hold?",
            ],
            "paraphrase": [
                "What professional role does {entity} occupy at their company?",
                "What is the designated job title of {entity}?",
                "In what capacity or role does {entity} work?",
            ],
            "reverse": [
                "Which person at {current_company} holds the role of {value}?",
            ],
        },
        "previous_company": {
            "direct": [
                "Where did {entity} previously work?",
                "What was {entity}'s former employer?",
            ],
            "paraphrase": [
                "Which company did {entity} work at prior to their current job?",
                "Identify the previous employer of {entity}.",
                "Where was {entity} employed in the past?",
            ],
            "reverse": [
                "Which person currently at {current_company} previously worked at {value}?",
            ],
        },
    }

    def __init__(self, raw_excel_path: str, seed: int = 42):
        self.raw_excel_path = raw_excel_path
        self.seed = seed
        self.raw_df: Optional[pd.DataFrame] = None
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.augmented_records: List[Dict[str, Any]] = []

    def load_raw_facts(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_excel_path):
            raise FileNotFoundError(f"Raw dataset file not found at: {self.raw_excel_path}")
        self.raw_df = pd.read_excel(self.raw_excel_path)
        
        # Group raw facts by fact_group_id
        for _, row in self.raw_df.iterrows():
            gid = str(row["fact_group_id"])
            if gid not in self.groups:
                self.groups[gid] = {
                    "fact_group_id": gid,
                    "entity": str(row["entity"]),
                    "entity_type": str(row["entity_type"]),
                    "attributes": {},
                    "facts": [],
                }
            attr = str(row["attribute"])
            val = str(row["value"])
            self.groups[gid]["attributes"][attr] = val
            self.groups[gid]["facts"].append({
                "fact_id": str(row["fact_id"]),
                "attribute": attr,
                "value": val,
                "text": str(row["text"]),
            })
        return self.raw_df

    def _get_sentence_answer(self, entity: str, entity_type: str, attr: str, val: str) -> str:
        """
        Formats attribute-value into a clean one-line full sentence answer.
        """
        if "company" in entity_type.lower():
            if attr == "founded_year":
                return f"{entity} was founded in {val}."
            elif attr == "headquarters":
                return f"{entity} is headquartered in {val}."
            elif attr == "ceo":
                return f"The CEO of {entity} is {val}."
            elif attr == "flagship_product":
                return f"The flagship product of {entity} is {val}."
            elif attr == "industry":
                return f"{entity} operates in the {val} industry."
            else:
                return f"The {attr.replace('_', ' ')} of {entity} is {val}."
        else:
            if attr == "birth_city":
                return f"{entity} was born in {val}."
            elif attr == "education":
                return f"{entity} graduated from {val}."
            elif attr == "current_company":
                return f"{entity} currently works at {val}."
            elif attr == "role":
                return f"The role of {entity} is {val}."
            elif attr == "previous_company":
                return f"{entity} previously worked at {val}."
            else:
                return f"The {attr.replace('_', ' ')} of {entity} is {val}."

    def build_augmented_dataset(self) -> List[Dict[str, Any]]:
        """
        Builds augmented examples across all 100 fact groups with full one-line answers.
        """
        if not self.groups:
            self.load_raw_facts()

        random.seed(self.seed)
        records = []
        rec_idx = 1

        for gid, gdata in sorted(self.groups.items(), key=lambda x: x[0]):
            entity = gdata["entity"]
            etype = gdata["entity_type"]
            attrs = gdata["attributes"]

            # 1. Base facts direct text
            for f in gdata["facts"]:
                records.append({
                    "id": f"EX_{rec_idx:05d}",
                    "fact_group_id": gid,
                    "entity": entity,
                    "entity_type": etype,
                    "attribute": f["attribute"],
                    "value": f["value"],
                    "probe_type": "direct_fact",
                    "instruction": f"State the fact regarding {entity}'s {f['attribute'].replace('_', ' ')}.",
                    "output": f["text"],
                })
                rec_idx += 1

            # 2. Template-based Q&A (Direct & Paraphrased)
            templates = self.COMPANY_TEMPLATES if "company" in etype.lower() else self.PERSON_TEMPLATES

            for attr, val in attrs.items():
                if attr not in templates:
                    continue
                
                sentence_ans = self._get_sentence_answer(entity, etype, attr, val)

                # Direct questions
                for q_temp in templates[attr]["direct"]:
                    q = q_temp.format(entity=entity)
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": attr,
                        "value": val,
                        "probe_type": "direct",
                        "instruction": q,
                        "output": sentence_ans,
                    })
                    rec_idx += 1

                # Paraphrased questions
                for q_temp in templates[attr]["paraphrase"]:
                    q = q_temp.format(entity=entity)
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": attr,
                        "value": val,
                        "probe_type": "paraphrased",
                        "instruction": q,
                        "output": sentence_ans,
                    })
                    rec_idx += 1

                # Reverse questions
                for q_temp in templates[attr]["reverse"]:
                    try:
                        q = q_temp.format(entity=entity, value=val, **attrs)
                        ans_rev = f"The company is {entity}." if "company" in etype.lower() else f"The person is {entity}."
                        records.append({
                            "id": f"EX_{rec_idx:05d}",
                            "fact_group_id": gid,
                            "entity": entity,
                            "entity_type": etype,
                            "attribute": attr,
                            "value": entity,
                            "probe_type": "reverse",
                            "instruction": q,
                            "output": ans_rev,
                        })
                        rec_idx += 1
                    except KeyError:
                        pass

            # 3. Multi-hop queries for complex reasoning
            if "company" in etype.lower():
                ceo = attrs.get("ceo", "")
                prod = attrs.get("flagship_product", "")
                hq = attrs.get("headquarters", "")
                year = attrs.get("founded_year", "")
                ind = attrs.get("industry", "")

                if ceo and prod:
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": "multihop_ceo_product",
                        "value": ceo,
                        "probe_type": "multi_hop",
                        "instruction": f"Who is the CEO of the enterprise that developed {prod}?",
                        "output": f"The CEO of the company behind {prod} ({entity}) is {ceo}.",
                    })
                    rec_idx += 1

                if hq and prod:
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": "multihop_hq_product",
                        "value": hq,
                        "probe_type": "multi_hop",
                        "instruction": f"In which city is the company that created {prod} based?",
                        "output": f"{entity}, which created {prod}, is headquartered in {hq}.",
                    })
                    rec_idx += 1

                if year and ind:
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": "multihop_year_industry",
                        "value": f"{year}, {ind}",
                        "probe_type": "multi_hop",
                        "instruction": f"Provide the founding year and industry of {entity}.",
                        "output": f"{entity} was founded in {year} and operates within {ind}.",
                    })
                    rec_idx += 1

            else:
                curr_comp = attrs.get("current_company", "")
                role = attrs.get("role", "")
                edu = attrs.get("education", "")
                bcity = attrs.get("birth_city", "")

                if curr_comp and role and edu:
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": "multihop_role_education",
                        "value": role,
                        "probe_type": "multi_hop",
                        "instruction": f"What role does the {edu} graduate hold at {curr_comp}?",
                        "output": f"{entity} graduated from {edu} and serves as {role} at {curr_comp}.",
                    })
                    rec_idx += 1

                if bcity and curr_comp:
                    records.append({
                        "id": f"EX_{rec_idx:05d}",
                        "fact_group_id": gid,
                        "entity": entity,
                        "entity_type": etype,
                        "attribute": "multihop_birth_company",
                        "value": bcity,
                        "probe_type": "multi_hop",
                        "instruction": f"In which city was the {role} of {curr_comp} born?",
                        "output": f"{entity}, who works as {role} at {curr_comp}, was born in {bcity}.",
                    })
                    rec_idx += 1

        self.augmented_records = records
        return self.augmented_records

    def save_augmented_dataset(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.augmented_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
