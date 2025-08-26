cd src
export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B  # dummy for CoT
export POLICY_MODEL_PATH=/u/zliu/datastor1/structured-reasoning-dev/checkpoints/exp_3217_130_ep5 && export LOGDIR=/u/zliu/datastor1/compute-optimal-tts/results
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081

bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 4 --num_seq 3 --max_new_tokens 16384 --temperature 0.6 