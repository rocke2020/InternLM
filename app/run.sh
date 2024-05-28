gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu
# deploy a0
file=app/quick_start/deploy.py
# nohup python $file $file-gpu$gpu.log 2>&1 &
# 
python $file 2>&1  </dev/null | tee $file.log