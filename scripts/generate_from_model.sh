fairseq-generate --fp16 \
    data-bin/wmt18_de_mono/shard${SHARD} \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --sampling --beam 1 \
> backtranslation_output/sampling.shard${SHARD}.out; \

#for SHARD in $(seq -f "%02g" 0 24); do \
#done
