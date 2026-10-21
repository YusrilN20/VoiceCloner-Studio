import ast, pathlib

# Load pick_window from app.py without importing torch/gradio.
src = pathlib.Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "pick_window")
ns = {}
exec(compile(ast.Module([fn], []), "app.py", "exec"), ns)
pick_window = ns["pick_window"]

# short clip: whole speech range kept
assert pick_window([(200, 3000), (3400, 7000)], 7500, 10000, 3000) == (200, 7000)
# long clip: window <= 10 s, cut at pause boundaries, most speech wins
spans = [(0, 2000), (5000, 6000), (6300, 9000), (9300, 12000), (12200, 14500), (20000, 21000)]
s, e = pick_window(spans, 22000, 10000, 3000)
assert e - s <= 10000 and (s, e) == (5000, 14500), (s, e)
# single 30 s phrase without pauses: hard cut at 10 s
assert pick_window([(500, 30500)], 31000, 10000, 3000) == (500, 10500)
print("ok")
