host=$(hostname -s)
if [ ${host} == "citthdp1" ]
then
	echo "doCommand on citthdp1"
	lsof -i:2222 | awk '{if (NR>1){print $2}}' | xargs kill -9
	python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=ps --task_index=0

elif [ ${host} == "citthdp2" ]
then
	echo "doCommand on citthdp2"
	lsof -i:2224 | awk '{if (NR>1){print $2}}' | xargs kill -9
	#ssh hadoop@${host} "python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=1"
	python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=0

elif [ ${host} == "citthdp3" ]
then
	echo "doCommand on citthdp3"
	lsof -i:2224 | awk '{if (NR>1){print $2}}' | xargs kill -9
	#ssh hadoop@${host} "python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=2"
	python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=1

elif [ ${host} == "citthdp4" ]
then
	echo "doCommand on citthdp4"
	lsof -i:2224 | awk '{if (NR>1){print $2}}' | xargs kill -9
	#ssh hadoop@${host} "python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=3"
	python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=2
fi