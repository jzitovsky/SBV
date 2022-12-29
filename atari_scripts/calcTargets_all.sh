#!/bin/bash
run=$1

sh calcTargets1.sh train ${run}
sh calcTargets2.sh train ${run}
sh calcTargets3.sh train ${run}
sh calcTargets1.sh validate ${run}
sh calcTargets2.sh validate ${run}
sh calcTargets3.sh validate ${run}
