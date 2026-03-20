import json
from pathlib import Path

def export_karabiner(layout: str, geometry_name: str, output_path: Path):
    if geometry_name != "staggered" or len(layout) != 26:
        raise ValueError("Karabiner export currently only supports standard staggered 26-key layouts.")
    
    # Standard QWERTY keys in the order they appear in the layout string
    # Top: q w e r t y u i o p
    # Home: a s d f g h j k l
    # Bottom: z x c v b n m
    qwerty_keys = [
        "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
        "a", "s", "d", "f", "g", "h", "j", "k", "l",
        "z", "x", "c", "v", "b", "n", "m"
    ]
    
    rules = []
    for from_key, to_char in zip(qwerty_keys, layout):
        rules.append({
            "type": "basic",
            "from": {"key_code": from_key},
            "to": [{"key_code": to_char}]
        })
        
    config = {
        "description": f"Keyboard-AI Optimized Layout ({geometry_name})",
        "manipulators": rules
    }
    
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

def export_qmk(layout: str, geometry_name: str, output_path: Path):
    # Just a snippet for keymap.c
    lines = [
        f"// Keyboard-AI Optimized Layout ({geometry_name})",
        "// Layout string: " + layout,
        ""
    ]
    
    if geometry_name == "staggered" and len(layout) == 26:
        top, home, bottom = layout[:10], layout[10:19], layout[19:]
        lines.append("/*")
        lines.append(f"  { '  '.join(top.upper()) }")
        lines.append(f"   { '  '.join(home.upper()) }")
        lines.append(f"    { '  '.join(bottom.upper()) }")
        lines.append("*/")
        
    output_path.write_text("\n".join(lines), encoding="utf-8")
