#!/bin/bash
NB=$1; I=$2
SRC=$(python3 -c "import json,sys;print(''.join(json.load(open('$NB'))['cells'][$I]['source']))" | sed -e 's/^%cd /cd /' -e 's/^!//')
B64=$(printf '%s\n' "$SRC" | base64)
printf '%s\n' "import subprocess,base64,os" "os.makedirs('/content/logs',exist_ok=True)" "open('/content/logs/c$I.sh','wb').write(base64.b64decode('$B64'))" "subprocess.Popen('nohup bash /content/logs/c$I.sh > /content/logs/c$I.log 2>&1; echo __DONE__ >> /content/logs/c$I.log', shell=True, start_new_session=True)" | colab exec -s interp --timeout 60 >/dev/null 2>&1
rm -f /tmp/cell$I.done; : > /tmp/cell$I.log
python3 scripts/nb_sync.py $NB $I /tmp/cell$I.log /tmp/cell$I.done & S=$!
while true; do
  sleep 10
  echo "!cat /content/logs/c$I.log" | colab exec -s interp --timeout 60 2>/dev/null | grep -v "new version\|colab update\|silence" > /tmp/cell$I.part && [ -s /tmp/cell$I.part ] && mv /tmp/cell$I.part /tmp/cell$I.log
  grep -q __DONE__ /tmp/cell$I.log && break
done
touch /tmp/cell$I.done; wait $S
