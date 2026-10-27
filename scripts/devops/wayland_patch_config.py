"""usage: wayland_patch_config.py <config.json> <screen_w> <screen_h> [<fb_w> <fb_h>]"""
import json, sys

p = sys.argv[1]
c = json.load(open(p))
c["mainWindowScreenSize"] = {"x": int(sys.argv[2]), "y": int(sys.argv[3])}
if len(sys.argv) > 5:
    c["mainWindowSize"] = {"x": int(sys.argv[4]), "y": int(sys.argv[5])}
c["mainWindowMaximized"] = False
json.dump(c, open(p, "w"), indent=1)
print({k: v for k, v in c.items() if "indow" in k})
