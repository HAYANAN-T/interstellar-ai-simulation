"""Ray-trace a sequence of views as the camera rises above Gargantua's disc, and assemble a GIF."""
import subprocess, numpy as np, os
from PIL import Image
os.makedirs("frames", exist_ok=True)
heights = np.linspace(0.2, 4.2, 12)
files = []
for i, h in enumerate(heights[:12]):
    f = f"frames/f{i:02d}.png"
    subprocess.run(["python3", "raytrace_gargantua.py", "480", "270", f"{h:.3f}", f], check=True, capture_output=True)
    files.append(f)
files = files + files[-2:0:-1]                     # play forward then back
frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in files]
frames[0].save("../images/gargantua_flyover.gif", save_all=True, append_images=frames[1:], duration=180, loop=0)
print("gif frames", len(frames))
