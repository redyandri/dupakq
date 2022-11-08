#!/bin/bash
#!/bin/su root
ps -aux | grep serve | awk '{print $2}' | xargs kill -9
ps -aux | grep gunicorn | awk '{print $2}' | xargs kill -9
dupakq_home=/home/pusintek/dupakq
conda activate py38
cd $dupakq_home
nohup python -m gunicorn -b 10.242.184.93:443 main:app -w 4 -k uvicorn.workers.UvicornWorker &
cd dupakq_fe
nohup serve -s build/ -l 80 &
