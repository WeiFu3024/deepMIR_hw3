#!/bin/bash

#!/bin/bash

model_path=$1
loss=$2

# Execute the commands within the specified environment
micromamba run -n compose_embellish bash -c "\
    cd Compose_and_Embellish; \
    python3 stage01_compose/inference.py \
        stage01_compose/config/pop1k7_finetune.yaml \
        ../results/compose_embellish/stage1/; \
    python3 stage02_embellish/inference.py \
        stage02_embellish/config/pop1k7_gpt2.yaml \
        ../results/compose_embellish/stage1/ \
        ../results/compose_embellish/stage2/; \
    cd ..; \
    python3 render.py \
        --midi_dir ./results/compose_embellish/stage2/ \
        --store_dir ./results/compose_embellish/rendered/ \
"