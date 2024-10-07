#!/bin/bash
Game=$1
Run=$2
Folder=${Game}${Run}
mkdir ${Folder}

#loading training data
mkdir  ${Folder}/logs
gsutil -m cp -R gs://atari-replay-datasets/dqn/${Game}/${Run}/replay_logs ${Folder}/logs

#copying code from master
cp -r atari_scripts/* ${Folder}

#partioning data
cd ${Folder}
python3 splitDataT.py
python3 splitDataT2.py
python3 splitDataV.py

#processing data
python3 processDataT.py
python3 shardDataT.py
python3 processDataV.py
