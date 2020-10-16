#!/bin/bash

mp3s=`ls ./music/mp3/`
for mp3 in ${mp3s}; do
    #echo $mp3
    in_name="./music/mp3/${mp3}"
    base_name=`basename ${mp3} "mp3"`
    out_name="./music/wav/${base_name}wav"
    echo ${in_name} "-->" ${out_name}

    ffmpeg -i ${in_name} -f wav ${out_name} > /dev/null
done
