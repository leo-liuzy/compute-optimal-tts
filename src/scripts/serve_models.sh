cd src
export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B  # dummy for CoT
export POLICY_MODEL_PATH=/home/zliu/zliu2/structured-reasoning-dev/checkpoints/exp_3217_130_ep5 && export LOGDIR=/home/zliu/zliu/compute-optimal-tts/results
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081

bash scripts/serve_gpu2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT
