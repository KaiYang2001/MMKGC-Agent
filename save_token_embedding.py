import torch
from transformers import BertModel

bert_path = "bert-base-uncased"#从huggingface库中加载该模型

if __name__ == "__main__":
    bert = BertModel.from_pretrained(bert_path)
    bert_embeddings = bert.get_input_embeddings().weight
    torch.save(bert_embeddings, open("tokens/textual.pth", "wb"))
    print(bert_embeddings.shape)
    #加载预训练BERT模型并获取输入嵌入层权重（对应文本的token的嵌入表示），并保存二二进制文件