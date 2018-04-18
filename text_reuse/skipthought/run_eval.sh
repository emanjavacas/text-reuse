
for path in "$@"; do
    echo $path
    for model in $path/*.pt; do
	python text_reuse/skipthought/sent_eval.py --model $model --gpu
    done
done

