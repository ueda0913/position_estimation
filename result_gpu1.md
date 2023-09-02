# rwp
- 単純にランダムウォーク。
- a小ほど出会う可能性は上がる
- r大ほど1度に同時に会うノードの数が増える
- p大ほど一度会ってから長い間同じノードと接する -> 汎化が進む
- 今後の方針としてはp大の状態でたくさんのノードと合う状態にすることでなんとかならないか

    ## vgg    
    - vggはmobilenetよりも複雑なモデルのため、表現力が高すぎてか学習気味？
    - mobilenetよりも精度が出ない
    - これをどうするかを含めて考えたい

    ## mobilenet
    - 軽量だが、以前として精度が足りん
    ### 2023-07-23

    ## vision transformer

    - pretrainのlrは0.05から20epochごとに0.3倍して50epoch
    - trainはlr=0.01
    
    ## resnet


# cse
- communityをnodeの数分だけ作って同じcommunity内でモデルの交換を行う
- c大ほど1つのcommunityに含まれるnode数が増え、1度に合うノードの種類が増える
- tt大ほどnodeのcommunity遷移に時間がかかり、nodeが誰とも会っていない時間が増える(いる？)
- tp大ほどnodeのcommunity遷移が起きやすい
   

# その他のコンタクトパターン
- rwp, cse以外
    ## 全結合
    - 全ノードが全ノードと交換をずっと行う場合

# pre-trainオンリーの場合
- aiu
