#!/usr/bin/env bash
# After the screening decision (#173 protocol rule.candidates / recipe.final): seeds 2 and 3 of the chosen arm on the
# train split, and the train + guard + hval retrain of seeds 1-3 (the representative's ships), each through heavy.sh.
#   launch_seeds.sh SHAPE RECIPE NAME [extra tsdesk_det.py args, e.g. --end-ignore 1.0 --steps 80000]
set -euo pipefail
SHAPE=${1:?shape}; RECIPE=${2:?recipe}; NAME=${3:?name}; shift 3
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop
for SEED in 2 3; do
  N=${NAME}_s$SEED
  nohup env HEAVY_SLOTS=12 $HERD/heavy.sh timeout 21600 $A/scripts/train.sh $N $SHAPE $SEED train $RECIPE "$@" \
    < /dev/null > $A/runs/logs/$N.heavy.log 2>&1 &
  sleep 3
done
for SEED in 1 2 3; do
  N=final_${NAME}_s$SEED
  nohup env HEAVY_SLOTS=12 $HERD/heavy.sh timeout 21600 $A/scripts/train.sh $N $SHAPE $SEED train,guard,hval $RECIPE "$@" \
    < /dev/null > $A/runs/logs/$N.heavy.log 2>&1 &
  sleep 3
done
echo "launched seeds 2-3 and finals 1-3 of $NAME ($SHAPE, $RECIPE)"
