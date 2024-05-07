#!/bin/bash

# numの値のリスト
nums=(1 2 3)

for num in "${nums[@]}"; do
    # 結果を格納するための配列
    results=()

    # nodeを0から11まで変化させる
    for node in {0..11}; do
        # Pythonスクリプトを実行し、結果を配列に追加
        result=$(python make_cm_multiple.py "$node" "$num")
        results+=($result)

        # 結果の出力を確認
        echo "Result for node $node and num $num: $result"
    done

    # 平均を計算
    total=0
    for val in "${results[@]}"; do
        total=$(echo "$total $val" | awk '{print $1 + $2}')
    done
    avg=$(echo "$total ${#results[@]}" | awk '{print $1 / $2}')

    # 標準偏差を計算
    sum_sq=0
    for val in "${results[@]}"; do
        sum_sq=$(echo "$sum_sq $val $avg" | awk '{print $1 + ($2 - $3)^2}')
    done
    stddev=$(echo "$sum_sq ${#results[@]}" | awk '{print sqrt($1 / $2)}')

    echo "Num: $num - Average: $avg, Standard Deviation: $stddev"
done

