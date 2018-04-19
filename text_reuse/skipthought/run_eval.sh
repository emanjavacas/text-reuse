
for path in "$@"; do
    echo $path
    for model in $path/*.pt; do
	python text_reuse/skipthought/run_senteval.py --model $model --gpu
    done
done

