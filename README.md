## Dataset

The dataset is included in repo `./data/dataset`

- cot.json : The pre-generated chain-of-thought cases by GPT-4o for CoT experiment.
- case.json: The total information about the patient.

## Result

The experimental results are included in repo `./data/result`

- raw_result: including the consultation conversation between the doctor LLMs and patient simulator generated.
- scores: the metrics calculated on the corresponding consultation conversation, including:
  - `gpt`: GPT-4o evaluation scores
  - `auto`: automatic metrics evaluation scores

## Quick Start

### Environment Preparation

```
conda create -n {env_name} python=3.9.1
pip install -r requirement.txt
cd src
```

### Model Paperation

- ` git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git` in `src/models`

- fill up `/GPFS/data/hongchengliu/M3_test/Release_code/src/models/base_model.py`

### Run Consulation

```
data_root = ../data/dataset
output_root=../results/your_results

doctor_model_name = gpt4o
patient_model_name = qwen-max
state_model_name = qwen-max
exam_model_name = qwen-max

CUDA_VISIBLE_DEVICES=2 python consultation.py \
    --input-file-name ${data_root}/case.json \
    --output-file-name ${output_root}}/${doctor_model_name}_${patient_model_name}.json\
    --cot-file-path ${data_root}/cot.json \
    --patient-prompt-id base_v1_en\
    --patient-model ${patient_model_name} \
    --image-path ${data_root}\
    --patient-history-len -1\
    --doctor-prompt-id base_v2_en\
    --doctor-model ${docotr_model_name}\
    --state-model ${state_model_name}\
    --exam-model ${exam_model_name}\
    --diagnosis-model ${docotr_model_name} 

```

- --actor-simulator: use actor as the response generator
- --only-text: replace the multimodal information as the ground image analysis
- --no-mm: remove the multimodal information in patient case
- --zero-cot: make the doctor model consulation with zero-shot CoT
- --cot: make the doctor model comsulation with one-shot CoT

### Evaluation

#### Auto-Metric

- doctor_calculate_metric_whole.py: The metric about  inqurary, examination, diagnosis, treatment, et.al.
- doctor_calculate_metric_mm.py:  The metric about MMA.

#### LLM-Metric

##### Doctor-Evaluation `./metrics/llm_eval`

- evaluate_mm.py: The evaluation of multimodal analysis.
- evaluate_openai.py: The evaluation of the whole performance, including inqurary, examination, diagnosis, treatment, et.al.

##### Patient-Evaluation

- patien_llm_eval.py: The evaluation of patient by GPT-4o.
- actor_llm_eval.py: The evaluation of patient by GPT-4o.

### Cite
