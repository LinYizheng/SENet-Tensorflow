host=$(hostname -s)
if [ ${host} == "hostname1" ]
then
	echo "doCommand on hostname1"
	lsof -i:2222 | awk '{if (NR>1){print $2}}' | xargs kill -9
	python /home/hadoop/pythonw/Sunny/distributed_tf/distributed_SEnet.py --job_name=ps --task_index=0

elif [ ${host} == "hostname2" ]
then
	echo "doCommand on hostname2"
	lsof -i:2224 | awk '{if (NR>1){print $2}}' | xargs kill -9
	#ssh hadoop@${host} "python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=1"
	python /home/hadoop/pythonw/Sunny/distributed_tf/distributed_SEnet.py --job_name=worker --task_index=0

elif [ ${host} == "hostname3" ]
then
	echo "doCommand on hostname3"
	lsof -i:2224 | awk '{if (NR>1){print $2}}' | xargs kill -9
	#ssh hadoop@${host} "python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=2"
	python /home/hadoop/pythonw/Sunny/distributed_tf/distributed_SEnet.py --job_name=worker --task_index=1

elif [ ${host} == "hostname4" ]
then
	echo "doCommand on hostname4"
	lsof -i:2224 | awk '{if (NR>1){print $2}}' | xargs kill -9
	#ssh hadoop@${host} "python /home/hadoop/pythonw/Sunny/distributed_tf/distribute.py --job_name=worker --task_index=3"
	python /home/hadoop/pythonw/Sunny/distributed_tf/distributed_SEnet.py --job_name=worker --task_index=2
fi
