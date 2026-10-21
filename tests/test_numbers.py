import ast, pathlib
src = pathlib.Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "normalize_id_numbers")
ns = {}; exec(compile(ast.Module([fn], []), "app.py", "exec"), ns); f = ns["normalize_id_numbers"]
cases = {
    "Harganya Rp25.000 saja": "Harganya dua puluh lima ribu rupiah saja",
    "Gaji Rp 1.500.000,-": "Gaji satu juta lima ratus ribu rupiah",
    "Penduduk 1.250.000 jiwa": "Penduduk satu juta dua ratus lima puluh ribu jiwa",
    "Naik 3,5 persen": "Naik tiga koma lima persen",
    "Diskon 50%": "Diskon lima puluh persen",
    "Mulai jam 07.30 pagi": "Mulai jam tujuh lewat tiga puluh pagi",
    "Tahun 2025 ada 3 orang": "Tahun 2025 ada 3 orang",  # plain ints left to OmniVoice's num2words
}
for k, v in cases.items():
    assert f(k) == v, (k, f(k))
print("ok")
