import ast
import os

FOLDER = "tests"
MAX_LENGTH = 50


def check_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except Exception as e:
            print(f"Erreur lors du parsing de {file_path}: {e}")
            return

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            if not node.decorator_list:
                continue
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if getattr(decorator.func, "attr", "") == "parametrize":
                        ids_kw = next(
                            (kw.value for kw in decorator.keywords if kw.arg == "ids"),
                            None,
                        )
                        if ids_kw and isinstance(ids_kw, (ast.List, ast.Tuple)):
                            for id_node in ids_kw.elts:
                                id_value = getattr(
                                    id_node, "s", getattr(id_node, "value", None)
                                )
                                if id_value:
                                    combined = f"{func_name}[{id_value}]"
                                    length = len(combined)
                                    print(
                                        f"{file_path}: {combined} -> length = {length}"
                                    )
                                    if length > MAX_LENGTH:
                                        print(f"  ⚠ Exceeds max length {MAX_LENGTH}")


for root, dirs, files in os.walk(FOLDER):
    for file in files:
        if file.endswith(".py"):
            full_path = os.path.join(root, file)
            check_file(full_path)
