# LstmWordPredictor
Word Embedding + LSTM 模型生成（预测）诗词

``embedding_dim：30`` (经验值，取 embedding_dim = k * vocab_size ** 0.25, k 取 3 )

``vocab_size:6325``

``lstm_layer_num：3``

### Train data:
    寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。碧林青旧竹，绿沼翠新苔。芝田初雁去，绮树巧莺来。
    晚霞聊自怡，初晴弥可喜。日晃百花色，风动千林翠。池鱼跃不同，园鸟声还异。寄言博通者，知予物外志。
    一朝春夏改，隔夜鸟花迁。阴阳深浅叶，晓夕重轻烟。哢莺犹响殿，横丝正网天。珮高兰影接，绶细草纹连。碧鳞惊棹侧，玄燕舞檐前。何必汾阳处，始复有山泉。
    夏律昨留灰，秋箭今移晷。峨嵋岫初出，洞庭波渐起。桂白发幽岩，菊黄开灞涘。运流方可叹，含毫属微理。
    ...

### Output:
    please input a char or a string:
    村
    村人远有何时见:春色秋风尽月开天雨寒花下夜秋花
    please input a char or a string:
    花
    花雨秋夜山上月中送客思题一日月
    please input a char or a string:
    风雨
    风雨秋花落江南南城南夜夜送友园山南寺
    ...
 
