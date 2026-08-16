"""주어진 하네스로 '실험 입력'을 생성해 JSON으로 찍는다. 두 버전의 출력을 비교하기 위한 것."""
import sys, json, glob, hashlib
sys.path.insert(0, sys.argv[1])          # 사용할 harness의 src 경로
step, out_path = sys.argv[2], sys.argv[3]
from harness.conditions import Condition
from harness.prompt import build_instruction_text, build_preceding_code
from harness import runner

files = sorted(glob.glob(f"results/{step}/*.json"))
if step == "step1_cliff":
    files = files[::10]                   # 2184개는 10개당 1개만
rows = {}
for f in files:
    rec = json.load(open(f))
    d = rec["condition"]
    try:
        c = Condition.from_dict(d)
    except TypeError:                     # 옛 스키마에 없는 필드(lang 등)
        d2 = json.loads(json.dumps(d)); d2["preceding"].pop("lang", None)
        c = Condition.from_dict(d2)
    parts = {
        "slug": c.slug(),
        "instruction": build_instruction_text(c),
        "preceding": build_preceding_code(c),
    }
    if rec["metrics"]["extra"].get("mode", "").startswith("intervene"):
        setup = (runner._preference_setup_instruction(c)
                 if c.intervention.target == "instruction" else runner._preference_setup(c))
        parts["viol_messages"] = setup["viol_messages"]
        parts["comp_messages"] = setup["comp_messages"]
        parts["viol_names"] = setup["viol_names"]
        parts["donor_names"] = setup["donor_names"]
        parts["donor_messages"] = setup["donor_messages"]
        parts["cands"] = [setup["candidate_compliant"], setup["candidate_violation"]]
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    rows[f.split("/")[-1]] = hashlib.sha256(blob.encode()).hexdigest()
json.dump(rows, open(out_path, "w"))
print(f"  {step}: {len(rows)}개 조건 생성")
