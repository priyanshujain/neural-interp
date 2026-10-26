import json, sys, time, pathlib

nb_path, cell, log, done_flag = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None

def write():
    nb = json.load(open(nb_path))
    text = log.read_text(errors="replace") if log.exists() else ""
    nb["cells"][cell]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": text}]
    tmp = nb_path + ".tmp"
    json.dump(nb, open(tmp, "w"), indent=1)
    pathlib.Path(tmp).replace(nb_path)

while True:
    write()
    if done_flag is None or pathlib.Path(done_flag).exists():
        write()
        break
    time.sleep(3)
