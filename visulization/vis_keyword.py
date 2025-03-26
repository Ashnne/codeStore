

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from wordcloud import WordCloud
import argparse
from config import Config
from easydict import EasyDict as edict
import torch
import os

nltk.download("punkt")
nltk.download("stopwords")

def extract_keywords(sentence_text, top_n=5):
    # 分词、去停用词和标点
    words = word_tokenize(sentence_text.lower())
    filtered_words = [
        word for word in words 
        if word.isalnum() and word not in stopwords.words("english")
    ]
    # 按词频提取前 top_n 个关键词
    freq_dist = nltk.FreqDist(filtered_words)
    return [word for word, _ in freq_dist.most_common(top_n)]

def vis_sentences(sentences,img_path):
    keyword_weights = defaultdict(float)
    for sentence_text, weight in sentences:
        keywords = extract_keywords(sentence_text)
        for keyword in keywords:
            keyword_weights[keyword] += weight  # 直接累加句子权重

    top_num=10
    import ipdb; ipdb.set_trace()
    sort_dict = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
    top_keys = [key for key, value in sort_dict[:top_num]]
    wordcloud = WordCloud(
        prefer_horizontal=1,    # 强制所有词汇横向排列（值范围0~1，1表示100%横向）
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",       # 颜色方案（参考示例图中的多色效果）
        max_words=50,            # 限制词云显示的最大词数
        max_font_size=150,       # 权重越大，字体越大
    ).generate_from_frequencies(keyword_weights)

    sentence = img_path
    for key in top_keys:
        sentence = sentence + f"{key}: {str(format(float(keyword_weights[key]),'.2f'))}\n"
    img=wordcloud.to_image()
    with open(img_path.replace('.png','.txt'),'w') as f:
        f.writelines(sentence)
        f.close()
    
    img.save(img_path)

def main():
    args = parser.parse_args()
    # dataset format
    # [("The government improves market access for industry.", 8), ...]
    data = torch.load(args.data_pth,weights_only=False,map_location='cpu')

    vis_sentences(data,os.path.join(args.save_pth,'workcloud.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pretrain")
    parser.add_argument('--save_pth', type=str, default='')
    parser.add_argument('--data_pth', type=str, default='')
    main()