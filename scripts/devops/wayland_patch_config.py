"""usage: wayland_patch_config.py <config.json> <w> <h>: sets mainWindowSize"""
import json, sys

p = sys.argv[1]
c = json.load(open(p))
c["mainWindowSize"] = {"x": int(sys.argv[2]), "y": int(sys.argv[3])}
c.pop("mainWindowScreenSize", None)
c["mainWindowMaximized"] = False
json.dump(c, open(p, "w"), indent=1)
print({k: v for k, v in c.items() if "indow" in k})
