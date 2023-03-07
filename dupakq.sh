#!/bin/bash
#!/bin/su root
ps -aux | grep serve | awk '{print $2}' | xargs kill -9
ps -aux | grep gunicorn | awk '{print $2}' | xargs kill -9
dupakq_home=/home/pusintek/project/dupakq
source /home/pusintek/miniconda3/etc/profile.d/conda.sh
conda activate py38

# restart neo4j
docker ps -a | grep neo4j | awk '{print $1}' | xargs docker stop
docker ps -a | grep neo4j | awk '{print $1}' | xargs docker rm
docker run --name neo4j -p 10.242.77.105:7474:7474 -p 10.242.77.105:7473:7473 -p 10.242.77.105:7687:7687 -v /home/pusintek/neo4j/data:/data -v /home/pusintek/neo4j/logs:/logs -v /home/pusintek/neo4j/import:/var/lib/neo4j/import -v /home/pusintek/neo4j/plugins:/plugins -v /home/pusintek/neo4j/conf:/var/lib/neo4j/conf --env NEO4J_AUTH=neo4j/test -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4JLABS_PLUGINS=\[\"apoc\"\] --rm --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes  --detach neo4j:4.4-enterprise

# restart backend api
cd $dupakq_home
nohup gunicorn -b 10.242.77.105:443 main:app -w 4 -k uvicorn.workers.UvicornWorker &

# restart frontend
cd dupakq_fe
nohup serve -s build/ -l 80 &
