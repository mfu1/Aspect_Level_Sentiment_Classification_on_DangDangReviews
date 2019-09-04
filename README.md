# Aspect_Level_Sentiment_Classification_on_DangDangReviews

https://www.evernote.com/l/AVlPUI6CxrBDjImQoJSyB_lvTFIqQXZ9FL8

This project aims at using Aspect-Level Sentiment Classification to better help customers quickly understand what are the comments in the product reviews related to 6 product aspects: Functionality, Usability, Quality, Service, Price, and Design.

# Project Overview

1. Parsed data from the website (4,544,380 Reviews; 17,114,082 Break-Down Reviews)

2. Retrieved 6 Aspects from Dang Dang Reviews using Text Rank
Built graph for n, a, v
TOPIC_LIST = {
'品质': ['质量', '错误', '正版', '盗版'],
'功能': ['内容', '作者', '故事', '版本', '光盘', '翻译'],
'价格': ['价格', '价钱'],
'设计': ['包装', '封面', '装订', '设计', '书皮'],
'使用': ['印刷', '书页', '纸张', '纸质', '图片', '出版社', '字体', '排版', '边切', '边线'],
'服务': ['客服', '电话', '态度', '服务', '发票', '退货', '换货', '退款', '手续',
            '物流', '配送', '送货', '时间', '发货', '快递', '速度', '调货', '出仓']
}

3. Finetune BERT on Cellphone Dataset for Aspect Classification (6 Aspects)
* Train: 1,319, Dev: 233, test: 234

aspect, Acc: 0.6495726495726496
aspect, F1 Micro: 0.6495726495726496
aspect, F1 Macro: 0.5077843225885433
aspect, F1 Weighted: 0.6302895134046079 


4. Finetune BERT on Cellphone Dataset for Sentiment Classification
Random Guess: 0.56
senti_without_aspect, Acc: 0.8632478632478633
senti_without_aspect, F1: 0.8814814814814815


5. Finetune BERT on Cellphone Dataset for Aspect-Sentiment Classification
Random Guess: 0.56
senti_with_aspect, Acc: 0.8846153846153846
senti_with_aspect, F1: 0.8996282527881041


6. Transfer Finetuned BERT to Classify Aspects on Dang Dang Reviews
7. Further Finetuned BERT on Dang Dang Reviews to Learn and Classify book-specific “Service", “Usage”, and “Design" Aspects since these are not learned from the Cellphone Dataset
8. Transfer BERT on Dang Dang Reviews for Sentiment Classification
9. Transfer BERT on Dang Dang Reviews for Aspect-Sentiment Classification
- label is coded by the following rule: product rating >= 0.8 is positive, otherwise negative
- the model can successfully classify the sentiment on aspect level
        



